"""Коллаборативная фильтрация: implicit ALS (Hu, Koren, Volinsky, 2008) на numpy.

Реализация своя, а не из библиотеки implicit: данных мало (сотни пользователей,
тысячи объектов), а тяжёлая нативная зависимость усложняет сборку образа
на разных архитектурах (arm64 локально, x86_64 на ВМ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(slots=True)
class ALSModel:
    user_factors: np.ndarray  # (n_users, f)
    item_factors: np.ndarray  # (n_items, f)

    def scores(self, user_idx: int) -> np.ndarray:
        return self.item_factors @ self.user_factors[user_idx]


def _solve_side(
    interactions: sp.csr_matrix,
    fixed: np.ndarray,
    reg: float,
    alpha: float,
) -> np.ndarray:
    """Один шаг ALS: пересчёт факторов одной стороны при фиксированной другой."""
    n, f = interactions.shape[0], fixed.shape[1]
    out = np.zeros((n, f), dtype=np.float32)
    yty = fixed.T @ fixed
    eye = np.eye(f, dtype=np.float32)
    for i in range(n):
        start, end = interactions.indptr[i], interactions.indptr[i + 1]
        if start == end:
            continue
        idx = interactions.indices[start:end]
        conf = 1.0 + alpha * interactions.data[start:end]
        y = fixed[idx]
        a = yty + (y.T * (conf - 1.0)) @ y + reg * eye
        b = (y.T * conf).sum(axis=1)
        out[i] = np.linalg.solve(a, b)
    return out


def train_als(
    matrix: sp.csr_matrix,
    factors: int = 32,
    iterations: int = 10,
    reg: float = 0.1,
    alpha: float = 10.0,
    seed: int = 0,
) -> ALSModel:
    """Обучает ALS на бинарной матрице пользователи × объекты (лайки)."""
    rng = np.random.default_rng(seed)
    n_users, n_items = matrix.shape
    users = matrix.tocsr().astype(np.float32)
    items = users.T.tocsr()
    u = (rng.standard_normal((n_users, factors)) * 0.01).astype(np.float32)
    v = (rng.standard_normal((n_items, factors)) * 0.01).astype(np.float32)
    for _ in range(iterations):
        u = _solve_side(users, v, reg, alpha)
        v = _solve_side(items, u, reg, alpha)
    return ALSModel(user_factors=u, item_factors=v)


def top_n(scores: np.ndarray, n: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Индексы n лучших объектов, исключая заданные."""
    s = scores.astype(np.float32, copy=True)
    if exclude is not None and exclude.size:
        s[exclude] = -np.inf
    n = min(n, int(np.isfinite(s).sum()))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-s, n - 1)[:n]
    return part[np.argsort(-s[part])]
