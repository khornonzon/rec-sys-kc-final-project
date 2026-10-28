"""Каталог объектов в памяти: жанровые векторы и индексы.

Жанров мало (десятки), поэтому точный скоринг всего каталога в numpy
занимает доли миллисекунды и быстрее любого сетевого вызова.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Catalog:
    keys: list[str]
    vocab: list[str]
    matrix: np.ndarray  # (n_items, n_genres), строки L2-нормированы
    genre_sets: list[frozenset[str]]
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def rows(self, keys: set[str] | list[str]) -> np.ndarray:
        """Индексы строк для известных ключей (неизвестные пропускаются)."""
        return np.fromiter((self.index[k] for k in keys if k in self.index), dtype=np.int64)

    @classmethod
    def empty(cls) -> Catalog:
        return cls(keys=[], vocab=[], matrix=np.zeros((0, 0), dtype=np.float32), genre_sets=[])

    @classmethod
    def build(cls, items: dict[str, tuple[str, ...]], vocab: list[str]) -> Catalog:
        """Строит каталог из словаря key -> genres и словаря жанров."""
        gidx = {g: i for i, g in enumerate(vocab)}
        keys = list(items)
        matrix = np.zeros((len(keys), max(len(vocab), 1)), dtype=np.float32)
        genre_sets: list[frozenset[str]] = []
        for row, k in enumerate(keys):
            genres = items[k]
            genre_sets.append(frozenset(genres))
            for g in genres:
                j = gidx.get(g)
                if j is not None:
                    matrix[row, j] = 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        return cls(keys=keys, vocab=list(vocab), matrix=matrix, genre_sets=genre_sets)


def intra_list_diversity(genre_sets: list[frozenset[str]]) -> float:
    """1 − средний попарный Jaccard жанров в списке (метрика diversity)."""
    n = len(genre_sets)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = genre_sets[i], genre_sets[j]
            union = len(a | b)
            total += (len(a & b) / union) if union else 0.0
            pairs += 1
    return 1.0 - total / pairs
