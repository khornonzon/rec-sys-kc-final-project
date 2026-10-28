"""Qdrant: хранение ALS-векторов объектов и поиск ближайших к вектору пользователя."""

from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

NAMESPACE = uuid.UUID("5f0c5b6e-3b6f-4f2a-9c1a-7d3e2a8f1b2c")


def point_id(key: str) -> str:
    """Детерминированный UUID точки по ключу объекта."""
    return str(uuid.uuid5(NAMESPACE, key))


class QdrantStore:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def from_url(cls, url: str, collection: str) -> QdrantStore:
        return cls(AsyncQdrantClient(url=url, timeout=5), collection)

    async def close(self) -> None:
        await self.client.close()

    async def ping(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def exists(self) -> bool:
        return await self.client.collection_exists(self.collection)

    async def drop(self) -> None:
        if await self.exists():
            await self.client.delete_collection(self.collection)

    async def recreate(self, dim: int) -> None:
        await self.drop()
        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.DOT),
        )

    async def upsert_items(self, keys: list[str], vectors: np.ndarray, batch: int = 1000) -> None:
        for start in range(0, len(keys), batch):
            chunk_keys = keys[start : start + batch]
            chunk_vec = vectors[start : start + batch]
            await self.client.upsert(
                collection_name=self.collection,
                points=qm.Batch(
                    ids=[point_id(k) for k in chunk_keys],
                    vectors=chunk_vec.astype(np.float32).tolist(),
                    payloads=[{"item": k} for k in chunk_keys],
                ),
            )

    async def search(self, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        res = await self.client.query_points(
            collection_name=self.collection,
            query=vector.astype(np.float32).tolist(),
            limit=limit,
            with_payload=["item"],
        )
        return [(p.payload["item"], float(p.score)) for p in res.points if p.payload]
