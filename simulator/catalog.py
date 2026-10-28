"""Каталог для симуляции: реальные жанры из IMDB-выгрузки курса или синтетика."""

from __future__ import annotations

import random
from pathlib import Path

FALLBACK_GENRES = [
    "Drama", "Comedy", "Action", "Crime", "Adventure", "Thriller", "Romance", "Mystery",
    "Fantasy", "Animation", "Sci-Fi", "Horror", "Family", "Biography", "History", "War",
    "Music", "Documentary", "Western", "Sport", "Musical",
]  # fmt: skip


def load_catalog(path: str | None, n_items: int, seed: int) -> list[tuple[str, list[str]]]:
    """Возвращает список (item_id, genres). Пробует parquet, иначе генерирует."""
    rng = random.Random(seed)
    items: list[tuple[str, list[str]]] = []
    if path and Path(path).exists():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path, columns=["title_id", "genre"]).to_pylist()
            for row in table:
                genres = [g for g in (row["genre"] or []) if g]
                if genres:
                    items.append((str(row["title_id"]), genres))
        except Exception as e:
            print(f"Не удалось прочитать {path}: {e}; используется синтетический каталог")
            items = []
    if not items:
        for i in range(n_items):
            k = rng.choice([1, 2, 2, 3, 3])
            items.append((f"item_{i}", rng.sample(FALLBACK_GENRES, k)))
    rng.shuffle(items)
    return items[:n_items]
