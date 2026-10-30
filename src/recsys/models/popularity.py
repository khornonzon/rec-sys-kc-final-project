"""Популярность как доля лайков на показ (CTR) с жанровым априором.

Грейдер присылает фидбек не на каждый показ, а только на заранее оценённые объекты,
поэтому знаменателем служат показы, а не дизлайки. Априорный CTR объекта берётся
из статистики его жанров: так непоказанные объекты «нужных» жанров сразу получают
преимущество, а бандит быстро находит объекты, которые любят многие пользователи.
"""

from __future__ import annotations

import numpy as np


def global_ctr(likes: np.ndarray, shows: np.ndarray, prior_ctr: float = 0.01) -> float:
    """Сглаженная доля лайков на показ по всему каталогу."""
    return float((likes.sum() + 100.0 * prior_ctr) / (shows.sum() + 100.0))


def genre_prior(
    binary: np.ndarray,
    likes: np.ndarray,
    shows: np.ndarray,
    base_ctr: float,
    smoothing: float = 50.0,
) -> np.ndarray:
    """Априорный CTR каждого объекта — средний сглаженный CTR его жанров.

    binary — матрица объекты × жанры из 0/1. Объекты без жанров получают base_ctr.
    """
    g_likes = binary.T @ likes
    g_shows = binary.T @ shows
    g_ctr = (g_likes + smoothing * base_ctr) / (g_shows + smoothing)
    n_genres = binary.sum(axis=1)
    prior = (binary @ g_ctr) / np.maximum(n_genres, 1.0)
    prior[n_genres == 0] = base_ctr
    return prior.astype(np.float32)


def ctr_posterior_mean(
    likes: np.ndarray, shows: np.ndarray, prior: np.ndarray, strength: float
) -> np.ndarray:
    """Апостериорное среднее CTR: (likes + s·prior) / (shows + s).

    Для короткого горизонта (несколько запросов на пользователя, тысячи пользователей
    параллельно) жадное ранжирование по среднему точнее Thompson sampling: шум семпла при
    десятке показов на объект перекрывает разницу между объектами с одним и тремя лайками.
    """
    return ((likes + strength * prior) / (shows + strength)).astype(np.float32)


def exploration_bonus(shows: np.ndarray) -> np.ndarray:
    """Бонус редко показанным объектам (работает на покрытие каталога)."""
    return (1.0 / np.sqrt(1.0 + shows)).astype(np.float32)
