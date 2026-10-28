"""Формирование рекомендаций: кандидаты → гибридный скоринг → MMR → слоты исследования."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from recsys.config import Settings
from recsys.ids import from_json_literal
from recsys.metrics import CATALOG_SIZE, RECS_CF_USED, RECS_PROFILE_USED, SNAPSHOT_AGE
from recsys.models.genres import Catalog
from recsys.models.popularity import (
    ctr_posterior_mean,
    ctr_thompson,
    exploration_bonus,
    genre_prior,
    global_ctr,
    legacy_thompson,
)
from recsys.models.rerank import inject_exploration, minmax, mmr
from recsys.models.user_profile import build_profile
from recsys.storage.qdrant_store import QdrantStore
from recsys.storage.redis_store import RedisStore

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Snapshot:
    """Состояние каталога и счётчиков, обновляемое в фоне.

    Запрос пользователя не ходит за каталогом в Redis: он читает снапшот,
    которому не больше пары секунд.
    """

    epoch: int = 0
    catalog_version: int = 0
    catalog: Catalog = field(default_factory=Catalog.empty)
    orig: dict[str, str] = field(default_factory=dict)
    likes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dislikes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    shows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    prior: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    base_ctr: float = 0.01
    updated_at: float = 0.0


class SnapshotKeeper:
    def __init__(self, store: RedisStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.snapshot = Snapshot()
        self._lock = asyncio.Lock()

    async def refresh(self, force: bool = False) -> None:
        async with self._lock:
            snap = self.snapshot
            epoch = await self.store.get_epoch()
            version = await self.store.catalog_version()
            if force or epoch != snap.epoch or version != snap.catalog_version:
                version, vocab, items = await self.store.load_catalog()
                catalog = Catalog.build(items, vocab)
                orig = await self.store.load_all_orig()
                snap = Snapshot(epoch=epoch, catalog_version=version, catalog=catalog, orig=orig)
                CATALOG_SIZE.set(len(catalog))
            likes, dislikes, shows = await self.store.load_counters()
            snap.likes = _aligned(snap.catalog, likes)
            snap.dislikes = _aligned(snap.catalog, dislikes)
            snap.shows = _aligned(snap.catalog, shows)
            if len(snap.catalog):
                snap.base_ctr = global_ctr(snap.likes, snap.shows)
                snap.prior = genre_prior(
                    snap.catalog.binary,
                    snap.likes,
                    snap.shows,
                    snap.base_ctr,
                    self.settings.genre_prior_smoothing,
                )
            snap.updated_at = time.monotonic()
            self.snapshot = snap

    async def run(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception:
                log.exception("Не удалось обновить снапшот")
            SNAPSHOT_AGE.set(time.monotonic() - self.snapshot.updated_at)
            await asyncio.sleep(self.settings.snapshot_refresh_sec)


def _aligned(catalog: Catalog, counts: dict[str, int]) -> np.ndarray:
    arr = np.zeros(len(catalog), dtype=np.float32)
    for k, v in counts.items():
        row = catalog.index.get(k)
        if row is not None:
            arr[row] = v
    return arr


class Recommender:
    def __init__(
        self,
        store: RedisStore,
        qdrant: QdrantStore,
        keeper: SnapshotKeeper,
        settings: Settings,
    ) -> None:
        self.store = store
        self.qdrant = qdrant
        self.keeper = keeper
        self.s = settings
        self.rng = np.random.default_rng()

    async def recommend(self, user: str) -> list[object]:
        """Возвращает идентификаторы в исходном виде (как пришли в /add_items)."""
        state = await self.store.get_user_state(user)
        snap = self.keeper.snapshot
        if state.epoch != snap.epoch or state.catalog_version != snap.catalog_version:
            # Эпоха или каталог изменились через другой процесс — перечитываем синхронно,
            # не дожидаясь фонового обновления снапшота.
            await self.keeper.refresh(force=True)
            snap = self.keeper.snapshot
        catalog = snap.catalog
        if len(catalog) == 0:
            return []

        n = len(catalog)
        k = self.s.top_k
        excluded_rows = catalog.rows(state.excluded)
        liked_rows = catalog.rows(state.liked)
        disliked_rows = catalog.rows(state.disliked)

        # --- скоринг ---
        scores = np.zeros(n, dtype=np.float32)
        profile = build_profile(catalog.matrix, liked_rows, disliked_rows, self.s.dislike_weight)
        content = None
        if profile is not None:
            content = catalog.matrix @ profile
            scores += self.s.w_content * content
            RECS_PROFILE_USED.inc()

        # Популярность: CTR-бандит. Логарифм растягивает шкалу, чтобы объекты с CTR в разы выше
        # базового выигрывали заметно, а не упирались в один выброс при minmax.
        if self.s.pop_mode == "legacy":
            scores += self.s.w_pop * legacy_thompson(snap.likes, snap.dislikes, self.rng)
        else:
            if self.s.pop_sampling == "thompson":
                ctr = ctr_thompson(
                    snap.likes, snap.shows, snap.prior, self.s.ctr_prior_strength, self.rng
                )
            else:
                ctr = ctr_posterior_mean(
                    snap.likes, snap.shows, snap.prior, self.s.ctr_prior_strength
                )
                # Джиттер в доле базового CTR: разрывает ничьи на холодном старте (иначе все
                # пользователи получили бы один список), но не меняет порядок проверенных объектов.
                ctr = ctr + self.s.pop_jitter * snap.base_ctr * self.rng.random(n, dtype=np.float32)
            # Пол 1e-3: CTR ниже 0.1 % неразличимы, иначе редкие крошечные семплы растягивают шкалу
            # и съедают разницу между проверенным объектом и непоказанным.
            scores += self.s.w_pop * minmax(np.log(np.maximum(ctr, 1e-3)))
        scores += self.s.w_explore * exploration_bonus(snap.shows)

        cf_scores = await self._cf_scores(state, catalog)
        if cf_scores is not None:
            scores += self.s.w_cf * cf_scores
            RECS_CF_USED.inc()

        available = np.ones(n, dtype=bool)
        available[excluded_rows] = False
        n_available = int(available.sum())
        if n_available == 0:
            return []
        scores[~available] = -np.inf

        # --- кандидаты и MMR ---
        n_cand = min(self.s.n_candidates, n_available)
        cand = np.argpartition(-scores, n_cand - 1)[:n_cand]
        order = mmr(scores[cand], catalog.matrix[cand], k, self.s.mmr_lambda)
        selected = [int(cand[i]) for i in order]

        # --- слоты исследования (покрытие каталога) ---
        explore = self._exploration_pool(snap, available, content, selected)
        selected = inject_exploration(selected, explore, self.s.n_explore, k)

        keys = [catalog.keys[i] for i in selected]
        snap.shows[selected] += 1.0  # локально, до следующего обновления снапшота из Redis
        await self.store.record_shown(user, keys)
        return [_restore(snap.orig.get(key), key) for key in keys]

    async def _cf_scores(self, state, catalog: Catalog) -> np.ndarray | None:
        """Коллаборативная составляющая: ANN в Qdrant по вектору пользователя,
        при недоступности — предрассчитанный список из Redis."""
        n = len(catalog)
        if state.cf_vector is not None:
            try:
                hits = await self.qdrant.search(state.cf_vector, self.s.cf_candidates)
            except Exception:
                log.warning("Qdrant недоступен, используем предрассчитанный список", exc_info=True)
                hits = []
            if hits:
                rows = catalog.rows([h[0] for h in hits])
                raw = np.array([h[1] for h in hits if h[0] in catalog.index], dtype=np.float32)
                if rows.size:
                    out = np.zeros(n, dtype=np.float32)
                    out[rows] = minmax(raw)
                    return out
        if state.precomputed:
            rows = catalog.rows(state.precomputed)
            if rows.size:
                out = np.zeros(n, dtype=np.float32)
                out[rows] = np.linspace(1.0, 0.1, rows.size, dtype=np.float32)
                return out
        return None

    def _exploration_pool(
        self,
        snap: Snapshot,
        available: np.ndarray,
        content: np.ndarray | None,
        selected: list[int],
    ) -> list[int]:
        """Редко показанные объекты; при наличии профиля — не противоречащие ему."""
        n_explore = self.s.n_explore
        if n_explore <= 0:
            return []
        shows = snap.shows + self.rng.random(
            len(snap.shows), dtype=np.float32
        )  # случайный разрыв ничьих
        shows = shows.copy()
        shows[~available] = np.inf
        shows[selected] = np.inf
        if content is not None:
            shows[content < 0] = np.inf
        m = min(n_explore * 10, int(np.isfinite(shows).sum()))
        if m == 0:
            return []
        pool = np.argpartition(shows, m - 1)[:m]
        if content is not None:
            pool = pool[np.argsort(-content[pool])]
        else:
            self.rng.shuffle(pool)
        return [int(i) for i in pool[:n_explore]]


def _restore(orig: str | None, key: str) -> object:
    if orig is None:
        return key
    try:
        return from_json_literal(orig)
    except ValueError:
        return key
