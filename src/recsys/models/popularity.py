"""Популярность с исследованием: Thompson sampling по Beta(likes+1, dislikes+1)."""

from __future__ import annotations

import numpy as np


def thompson_scores(
    likes: np.ndarray, dislikes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Случайная выборка из апостериорного распределения доли лайков каждого объекта.

    Объекты без фидбека получают Beta(1, 1) = равномерное распределение, то есть
    случайный порядок — это и есть исследование на холодном старте. Для них берётся
    rng.random напрямую: beta-семплер в 30 раз дороже, а таких объектов большинство.
    """
    scores = rng.random(len(likes), dtype=np.float32)  # Beta(1, 1) — равномерное, дёшево
    with_feedback = (likes + dislikes) > 0
    if with_feedback.any():
        scores[with_feedback] = rng.beta(likes[with_feedback] + 1.0, dislikes[with_feedback] + 1.0)
    return scores


def exploration_bonus(shows: np.ndarray) -> np.ndarray:
    """Бонус редко показанным объектам (работает на покрытие каталога)."""
    return (1.0 / np.sqrt(1.0 + shows)).astype(np.float32)
