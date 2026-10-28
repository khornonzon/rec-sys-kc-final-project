"""Синтетические пользователи с латентными жанровыми предпочтениями."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class SimUser:
    user_id: str
    relevant: set[str]
    seen: set[str] = field(default_factory=set)

    def reaction(self, item: str) -> str:
        return "like" if item in self.relevant else "dislike"


def make_users(
    catalog: list[tuple[str, list[str]]],
    n_users: int,
    relevant_frac: float,
    seed: int,
) -> list[SimUser]:
    """Каждый пользователь любит 2–3 жанра; релевантны top-M объектов по полезности с шумом."""
    rng = random.Random(seed)
    genres = sorted({g for _, gs in catalog for g in gs})
    n_rel = max(10, int(len(catalog) * relevant_frac))
    users: list[SimUser] = []
    for u in range(n_users):
        favorites = rng.sample(genres, k=rng.choice([2, 3]))
        weights = {g: rng.uniform(0.5, 1.0) for g in favorites}
        utilities = []
        for item, gs in catalog:
            score = sum(weights.get(g, 0.0) for g in gs) / math.sqrt(len(gs))
            utilities.append((score + rng.gauss(0, 0.15), item))
        utilities.sort(reverse=True)
        users.append(
            SimUser(user_id=f"user_{seed}_{u}", relevant={i for _, i in utilities[:n_rel]})
        )
    return users
