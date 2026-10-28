"""Переранжирование: MMR для разнообразия и слоты исследования для покрытия."""

from __future__ import annotations

import numpy as np


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.ones_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def mmr(scores: np.ndarray, vectors: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance.

    На каждом шаге выбирается кандидат, максимизирующий
    lam * relevance − (1 − lam) * max_sim_to_selected.
    Возвращает индексы кандидатов в порядке выбора.
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    rel = minmax(scores)
    sims = vectors @ vectors.T  # косинусы, строки нормированы
    selected: list[int] = []
    max_sim = np.zeros(n, dtype=np.float32)
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        objective = lam * rel - (1.0 - lam) * max_sim
        objective[~available] = -np.inf
        j = int(np.argmax(objective))
        selected.append(j)
        available[j] = False
        max_sim = np.maximum(max_sim, sims[j])
    return selected


def inject_exploration(
    selected: list[int],
    explore_candidates: list[int],
    n_explore: int,
    k: int,
) -> list[int]:
    """Заменяет хвост списка объектами из пула исследования.

    Если список короче k, пул просто дополняет его. Дубликаты не добавляются.
    """
    if n_explore <= 0 or not explore_candidates:
        return selected[:k]
    chosen = set(selected)
    extras = [c for c in explore_candidates if c not in chosen][:n_explore]
    if not extras:
        return selected[:k]
    keep = max(0, k - len(extras))
    head = selected[:keep]
    return (head + extras)[:k]
