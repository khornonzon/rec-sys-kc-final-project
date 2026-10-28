"""Regular pipeline: потребитель очереди событий и периодическое обучение ALS.

Две корутины:
  * consumer — читает события из RabbitMQ, пишет их в Redis Stream `events`
    и идемпотентно применяет к множествам пользователя (второй, страхующий путь);
  * trainer — раз в TRAIN_INTERVAL_SEC обучает ALS на матрице лайков, кладёт
    векторы объектов в Qdrant, векторы пользователей и top-N списки — в Redis.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import scipy.sparse as sp
from prometheus_client import start_http_server

from recsys.config import Settings, get_settings
from recsys.logging_setup import setup_logging
from recsys.metrics import (
    PIPELINE_EVENTS,
    PIPELINE_LAST_LIKES,
    PIPELINE_TRAIN_SECONDS,
    PIPELINE_TRAININGS,
)
from recsys.models.cf import top_n, train_als
from recsys.schemas import Interaction
from recsys.storage.qdrant_store import QdrantStore
from recsys.storage.rabbit import consume
from recsys.storage.redis_store import RedisStore

log = logging.getLogger("pipeline")


async def consumer(store: RedisStore, settings: Settings) -> None:
    while True:
        try:
            async for payload in consume(settings.rabbitmq_url, settings.rabbitmq_queue):
                PIPELINE_EVENTS.inc()
                await store.append_event(payload)
                try:
                    ev = Interaction(
                        str(payload["user"]), str(payload["item"]), payload["reaction"]
                    )
                except (KeyError, TypeError):
                    log.warning("Событие без нужных полей: %r", payload)
                    continue
                if ev.reaction in ("like", "dislike"):
                    await store.apply_interaction_idempotent(ev)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Потребитель упал, переподключение через 5 с")
            await asyncio.sleep(5)


async def train_once(store: RedisStore, qdrant: QdrantStore, settings: Settings) -> bool:
    """Одно обучение. Возвращает True, если модель обновлена."""
    epoch = await store.get_epoch()
    total_likes = await store.total_likes()
    if total_likes < settings.min_likes_to_train:
        log.info("Лайков %d < %d, обучение пропущено", total_likes, settings.min_likes_to_train)
        return False

    _, _, items = await store.load_catalog()
    if not items:
        return False
    users = await store.all_users()
    feedback = await store.users_feedback(users)
    users = [u for u in users if feedback[u][0]]
    if len(users) < 2:
        return False

    item_keys = list(items)
    item_index = {k: i for i, k in enumerate(item_keys)}
    rows, cols = [], []
    for ui, u in enumerate(users):
        for key in feedback[u][0]:
            col = item_index.get(key)
            if col is not None:
                rows.append(ui)
                cols.append(col)
    if not rows:
        return False
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), len(item_keys))
    )

    started = time.perf_counter()
    model = await asyncio.to_thread(
        train_als,
        matrix,
        settings.als_factors,
        settings.als_iterations,
        settings.als_reg,
        settings.als_alpha,
    )
    PIPELINE_TRAIN_SECONDS.observe(time.perf_counter() - started)
    PIPELINE_LAST_LIKES.set(len(rows))

    if await store.get_epoch() != epoch:
        log.info("Эпоха сменилась во время обучения, результат отброшен")
        return False

    # Векторы объектов (только тех, у кого были лайки) — в Qdrant.
    liked_items = np.unique(cols)
    if not await qdrant.exists():
        await qdrant.recreate(settings.als_factors)
    await qdrant.upsert_items([item_keys[i] for i in liked_items], model.item_factors[liked_items])

    # Векторы пользователей и предрассчитанные списки — в Redis.
    user_vectors: dict[str, np.ndarray] = {}
    precomputed: dict[str, list[str]] = {}
    for ui, u in enumerate(users):
        liked, disliked, shown = feedback[u]
        exclude = np.fromiter(
            (item_index[k] for k in (liked | disliked | shown) if k in item_index), dtype=np.int64
        )
        best = top_n(model.scores(ui), settings.cf_top_n, exclude)
        user_vectors[u] = model.user_factors[ui]
        precomputed[u] = [item_keys[i] for i in best]
    await store.save_cf(user_vectors, precomputed, version=time.time_ns())
    PIPELINE_TRAININGS.inc()
    log.info(
        "ALS обучена: users=%d items_with_likes=%d likes=%d за %.1f с",
        len(users),
        len(liked_items),
        len(rows),
        time.perf_counter() - started,
    )
    return True


async def trainer(store: RedisStore, qdrant: QdrantStore, settings: Settings) -> None:
    while True:
        try:
            await train_once(store, qdrant, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Ошибка обучения")
        await asyncio.sleep(settings.train_interval_sec)


async def main() -> None:
    setup_logging()
    settings = get_settings()
    start_http_server(settings.pipeline_metrics_port)
    store = RedisStore.from_url(settings.redis_url)
    qdrant = QdrantStore.from_url(settings.qdrant_url, settings.qdrant_collection)
    log.info("Pipeline запущен, интервал обучения %.0f с", settings.train_interval_sec)
    try:
        await asyncio.gather(consumer(store, settings), trainer(store, qdrant, settings))
    finally:
        await qdrant.close()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
