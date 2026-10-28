"""Все ключи Redis и операции над ними в одном месте.

Схема ключей:
    epoch                      номер запуска грейдера; меняется в /cleanup
    catalog:version            растёт при каждом /add_items
    items:genres               HASH key -> JSON-список жанров
    items:orig                 HASH key -> исходный JSON-литерал идентификатора
    genres:vocab               LIST жанров в порядке первого появления
    item:likes|dislikes|shows  HASH key -> счётчик
    users:all                  SET всех пользователей
    user:{u}:shown|liked|disliked  SET объектов
    cf:version                 версия обученной ALS-модели
    cf:user_vec                HASH u -> float32 bytes вектора пользователя
    recs:{u}                   LIST предрассчитанных рекомендаций от pipeline
    events                     STREAM всех событий (лог для переобучения/аудита)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import orjson
from redis.asyncio import Redis

from recsys.schemas import Interaction, Item, Reaction


@dataclass(slots=True)
class UserState:
    epoch: int
    catalog_version: int
    shown: set[str]
    liked: set[str]
    disliked: set[str]
    cf_vector: np.ndarray | None
    precomputed: list[str]

    @property
    def excluded(self) -> set[str]:
        return self.shown | self.liked | self.disliked


class RedisStore:
    def __init__(self, redis: Redis) -> None:
        self.r = redis

    @classmethod
    def from_url(cls, url: str) -> RedisStore:
        return cls(Redis.from_url(url, decode_responses=False))

    async def close(self) -> None:
        await self.r.aclose()

    async def ping(self) -> bool:
        try:
            return bool(await self.r.ping())
        except Exception:
            return False

    # ---------- эпоха / очистка ----------

    async def get_epoch(self) -> int:
        v = await self.r.get("epoch")
        return int(v) if v else 0

    async def reset(self) -> int:
        """Полный сброс состояния. Возвращает новую эпоху."""
        await self.r.flushdb()
        epoch = time.time_ns()
        await self.r.set("epoch", epoch)
        return epoch

    # ---------- каталог ----------

    async def add_items(self, items: list[Item]) -> int:
        if not items:
            return 0
        vocab = await self.get_vocab()
        known = set(vocab)
        new_genres: list[str] = []
        for it in items:
            for g in it.genres:
                if g not in known:
                    known.add(g)
                    new_genres.append(g)
        pipe = self.r.pipeline(transaction=True)
        pipe.hset("items:genres", mapping={it.key: orjson.dumps(list(it.genres)) for it in items})
        pipe.hset("items:orig", mapping={it.key: it.orig for it in items})
        if new_genres:
            pipe.rpush("genres:vocab", *new_genres)
        pipe.incr("catalog:version")
        res = await pipe.execute()
        return int(res[-1])

    async def get_vocab(self) -> list[str]:
        raw = await self.r.lrange("genres:vocab", 0, -1)
        return [g.decode() for g in raw]

    async def catalog_version(self) -> int:
        v = await self.r.get("catalog:version")
        return int(v) if v else 0

    async def load_catalog(self) -> tuple[int, list[str], dict[str, tuple[str, ...]]]:
        pipe = self.r.pipeline(transaction=False)
        pipe.get("catalog:version")
        pipe.lrange("genres:vocab", 0, -1)
        pipe.hgetall("items:genres")
        version, vocab, genres = await pipe.execute()
        items = {k.decode(): tuple(orjson.loads(v)) for k, v in genres.items()}
        return int(version or 0), [g.decode() for g in vocab], items

    async def load_orig(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        vals = await self.r.hmget("items:orig", keys)
        return {k: v.decode() for k, v in zip(keys, vals, strict=True) if v is not None}

    async def load_all_orig(self) -> dict[str, str]:
        raw = await self.r.hgetall("items:orig")
        return {k.decode(): v.decode() for k, v in raw.items()}

    # ---------- счётчики популярности ----------

    async def load_counters(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        pipe = self.r.pipeline(transaction=False)
        pipe.hgetall("item:likes")
        pipe.hgetall("item:dislikes")
        pipe.hgetall("item:shows")
        likes, dislikes, shows = await pipe.execute()
        dec = lambda d: {k.decode(): int(v) for k, v in d.items()}  # noqa: E731
        return dec(likes), dec(dislikes), dec(shows)

    async def total_likes(self) -> int:
        vals = await self.r.hvals("item:likes")
        return sum(int(v) for v in vals)

    # ---------- состояние пользователя ----------

    async def get_user_state(self, user: str) -> UserState:
        pipe = self.r.pipeline(transaction=False)
        pipe.get("epoch")
        pipe.get("catalog:version")
        pipe.smembers(f"user:{user}:shown")
        pipe.smembers(f"user:{user}:liked")
        pipe.smembers(f"user:{user}:disliked")
        pipe.hget("cf:user_vec", user)
        pipe.lrange(f"recs:{user}", 0, -1)
        epoch, version, shown, liked, disliked, vec, pre = await pipe.execute()
        return UserState(
            epoch=int(epoch) if epoch else 0,
            catalog_version=int(version) if version else 0,
            shown={x.decode() for x in shown},
            liked={x.decode() for x in liked},
            disliked={x.decode() for x in disliked},
            cf_vector=np.frombuffer(vec, dtype=np.float32) if vec else None,
            precomputed=[x.decode() for x in pre],
        )

    async def record_shown(self, user: str, keys: list[str]) -> None:
        if not keys:
            return
        pipe = self.r.pipeline(transaction=False)
        pipe.sadd("users:all", user)
        pipe.sadd(f"user:{user}:shown", *keys)
        for k in keys:
            pipe.hincrby("item:shows", k, 1)
        await pipe.execute()

    async def record_interaction(self, ev: Interaction) -> None:
        """Лайк/дизлайк: множества пользователя и счётчики объекта.

        Если пользователь поменял мнение, объект переносится между множествами,
        а счётчик старой реакции уменьшается. Повторная реакция счётчик не меняет.
        """
        this, other = _sets_for(ev.reaction)
        moved = await self.r.smove(f"user:{ev.user}:{other}", f"user:{ev.user}:{this}", ev.item)
        pipe = self.r.pipeline(transaction=False)
        pipe.sadd("users:all", ev.user)
        if moved:
            pipe.hincrby(f"item:{COUNTER[other]}", ev.item, -1)
            pipe.hincrby(f"item:{COUNTER[this]}", ev.item, 1)
            await pipe.execute()
            return
        pipe.sadd(f"user:{ev.user}:{this}", ev.item)
        res = await pipe.execute()
        if res[-1] == 1:
            await self.r.hincrby(f"item:{COUNTER[this]}", ev.item, 1)

    async def apply_interaction_idempotent(self, ev: Interaction) -> None:
        """Идемпотентное применение события (без счётчиков) — путь через очередь."""
        this, other = _sets_for(ev.reaction)
        pipe = self.r.pipeline(transaction=False)
        pipe.sadd("users:all", ev.user)
        pipe.srem(f"user:{ev.user}:{other}", ev.item)
        pipe.sadd(f"user:{ev.user}:{this}", ev.item)
        await pipe.execute()

    async def append_event(self, payload: dict) -> None:
        await self.r.xadd(
            "events", {"data": orjson.dumps(payload)}, maxlen=1_000_000, approximate=True
        )

    # ---------- данные для обучения ----------

    async def all_users(self) -> list[str]:
        raw = await self.r.smembers("users:all")
        return sorted(x.decode() for x in raw)

    async def users_feedback(
        self, users: list[str]
    ) -> dict[str, tuple[set[str], set[str], set[str]]]:
        """Для каждого пользователя: (liked, disliked, shown)."""
        if not users:
            return {}
        pipe = self.r.pipeline(transaction=False)
        for u in users:
            pipe.smembers(f"user:{u}:liked")
            pipe.smembers(f"user:{u}:disliked")
            pipe.smembers(f"user:{u}:shown")
        res = await pipe.execute()
        out = {}
        for i, u in enumerate(users):
            liked, disliked, shown = res[3 * i], res[3 * i + 1], res[3 * i + 2]
            out[u] = (
                {x.decode() for x in liked},
                {x.decode() for x in disliked},
                {x.decode() for x in shown},
            )
        return out

    async def save_cf(
        self,
        user_vectors: dict[str, np.ndarray],
        precomputed: dict[str, list[str]],
        version: int,
    ) -> None:
        pipe = self.r.pipeline(transaction=True)
        pipe.delete("cf:user_vec")
        if user_vectors:
            pipe.hset(
                "cf:user_vec",
                mapping={u: v.astype(np.float32).tobytes() for u, v in user_vectors.items()},
            )
        for u, keys in precomputed.items():
            pipe.delete(f"recs:{u}")
            if keys:
                pipe.rpush(f"recs:{u}", *keys)
        pipe.set("cf:version", version)
        await pipe.execute()

    async def cf_version(self) -> int:
        v = await self.r.get("cf:version")
        return int(v) if v else 0


COUNTER = {"liked": "likes", "disliked": "dislikes"}


def _sets_for(reaction: Reaction) -> tuple[str, str]:
    """Имена множеств пользователя: (куда положить, откуда убрать)."""
    return ("liked", "disliked") if reaction == "like" else ("disliked", "liked")
