"""Метрики качества, совпадающие по смыслу с метриками грейдера."""

from __future__ import annotations

import math


def precision_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    return sum(r in relevant for r in recs[:k]) / k


def ndcg_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(recs[:k]) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def diversity(recs: list[str], genres: dict[str, list[str]]) -> float:
    """1 − средний попарный Jaccard жанров."""
    sets = [set(genres.get(r, [])) for r in recs]
    n = len(sets)
    if n < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            total += len(sets[i] & sets[j]) / len(union) if union else 0.0
            pairs += 1
    return 1.0 - total / pairs


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]
