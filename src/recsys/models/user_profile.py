"""Контентный профиль пользователя по жанрам лайкнутых и дизлайкнутых объектов."""

from __future__ import annotations

import numpy as np


def build_profile(
    matrix: np.ndarray,
    liked_rows: np.ndarray,
    disliked_rows: np.ndarray,
    dislike_weight: float = 0.5,
) -> np.ndarray | None:
    """Единичный вектор предпочтений или None, если фидбека ещё нет.

    Лайки притягивают к своим жанрам, дизлайки отталкивают с меньшим весом:
    дизлайк чаще означает «не тот фильм», чем «не тот жанр».
    """
    if liked_rows.size == 0 and disliked_rows.size == 0:
        return None
    profile = np.zeros(matrix.shape[1], dtype=np.float32)
    if liked_rows.size:
        profile += matrix[liked_rows].sum(axis=0)
    if disliked_rows.size:
        profile -= dislike_weight * matrix[disliked_rows].sum(axis=0)
    norm = float(np.linalg.norm(profile))
    if norm == 0.0:
        return None
    return profile / norm
