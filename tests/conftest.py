import fakeredis.aioredis
import numpy as np
import pytest

from recsys.config import Settings
from recsys.storage.redis_store import RedisStore


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, snapshot_refresh_sec=0.01, min_likes_to_train=5)


@pytest.fixture
async def store() -> RedisStore:
    r = fakeredis.aioredis.FakeRedis(decode_responses=False)
    s = RedisStore(r)
    yield s
    await s.close()


class FakeQdrant:
    """Стаб Qdrant: хранит векторы в памяти, ищет по скалярному произведению."""

    def __init__(self) -> None:
        self.vectors: dict[str, np.ndarray] = {}
        self.dim: int | None = None

    async def exists(self) -> bool:
        return self.dim is not None

    async def drop(self) -> None:
        self.vectors.clear()
        self.dim = None

    async def recreate(self, dim: int) -> None:
        self.vectors.clear()
        self.dim = dim

    async def upsert_items(self, keys, vectors, batch: int = 1000) -> None:
        for k, v in zip(keys, vectors, strict=True):
            self.vectors[k] = np.asarray(v, dtype=np.float32)

    async def search(self, vector, limit):
        scored = [(k, float(v @ vector)) for k, v in self.vectors.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    async def close(self) -> None:
        pass


@pytest.fixture
def qdrant() -> FakeQdrant:
    return FakeQdrant()
