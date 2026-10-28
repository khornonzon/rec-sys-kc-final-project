"""Синтетические пользователи, похожие на пользователей грейдера.

Наблюдения из реального прогона: у пользователя заранее оценено несколько сотен объектов,
фидбек приходит только на них (лайков втрое больше дизлайков), остальные показы остаются
без реакции. Популярность объектов имеет тяжёлый хвост, как в MovieLens.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class SimUser:
    user_id: str
    liked: set[str]
    disliked: set[str]
    seen: set[str] = field(default_factory=set)

    @property
    def relevant(self) -> set[str]:
        return self.liked

    def reaction(self, item: str) -> str | None:
        """like/dislike для оценённых объектов, None — реакции нет."""
        if item in self.liked:
            return "like"
        if item in self.disliked:
            return "dislike"
        return None


def make_users(
    catalog: list[tuple[str, list[str]]],
    n_users: int,
    rated_per_user: int,
    seed: int,
    like_share: float = 0.75,
    popularity_skew: float = 0.5,
    genre_strength: float = 1.5,
) -> list[SimUser]:
    """Каждый пользователь оценил `rated_per_user` объектов.

    Вероятность оценить объект пропорциональна его латентной популярности (Zipf с показателем
    popularity_skew) и exp(genre_strength · близость жанров к 2–3 любимым жанрам пользователя).
    Оценка — лайк с вероятностью, зависящей от той же близости и скрытого качества объекта.
    genre_strength калибруется так, чтобы контентный бейзлайн давал тот же выигрыш над случайным,
    что и на реальном грейдере (~5×).
    """
    rng = random.Random(seed)
    genres = sorted({g for _, gs in catalog for g in gs})
    n = len(catalog)
    ranks = list(range(1, n + 1))
    rng.shuffle(ranks)
    popularity = [1.0 / (r**popularity_skew) for r in ranks]
    quality = [rng.gauss(0.0, 1.0) for _ in range(n)]
    rated_per_user = min(rated_per_user, n)

    users: list[SimUser] = []
    for u in range(n_users):
        favorites = rng.sample(genres, k=rng.choice([2, 3]))
        weights = {g: rng.uniform(0.5, 1.0) for g in favorites}
        affinity = [
            sum(weights.get(g, 0.0) for g in gs) / math.sqrt(len(gs)) if gs else 0.0
            for _, gs in catalog
        ]
        exposure = [
            p * math.exp(genre_strength * a) for p, a in zip(popularity, affinity, strict=True)
        ]
        idx = _weighted_sample_without_replacement(rng, exposure, rated_per_user)
        liked, disliked = set(), set()
        for i in idx:
            logit = (
                math.log(like_share / (1 - like_share))
                + genre_strength * affinity[i]
                + 0.7 * quality[i]
            )
            p_like = 1.0 / (1.0 + math.exp(-logit))
            (liked if rng.random() < p_like else disliked).add(catalog[i][0])
        users.append(SimUser(user_id=f"user_{seed}_{u}", liked=liked, disliked=disliked))
    return users


def _weighted_sample_without_replacement(
    rng: random.Random, weights: list[float], k: int
) -> list[int]:
    """Efraimidis–Spirakis: k индексов без возвращения с вероятностями ∝ weights."""
    keys = [(rng.random() ** (1.0 / w) if w > 0 else 0.0, i) for i, w in enumerate(weights)]
    keys.sort(reverse=True)
    return [i for _, i in keys[:k]]
