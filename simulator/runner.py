"""Протокол грейдера: cleanup → add_items → раунды recs/interact с ограничением параллелизма."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from simulator.catalog import load_catalog
from simulator.metrics import diversity, ndcg_at_k, percentile, precision_at_k
from simulator.users import SimUser, make_users

TOP_K = 10


@dataclass
class RoundStats:
    precision: list[float] = field(default_factory=list)
    ndcg: list[float] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    short_lists: int = 0

    def summary(self) -> dict:
        n = max(1, len(self.precision))
        return {
            "requests": len(self.precision),
            "precision": sum(self.precision) / n,
            "ndcg": sum(self.ndcg) / n,
            "diversity": sum(self.diversity) / n,
            "latency_p50_ms": percentile(self.latency_ms, 0.5),
            "latency_p95_ms": percentile(self.latency_ms, 0.95),
            "short_lists": self.short_lists,
        }


@dataclass
class Config:
    recs_url: str = "http://localhost:5001"
    events_url: str = "http://localhost:5000"
    users: int = 300
    rounds: int = 6
    items: int = 3000
    catalog_path: str | None = None
    relevant_frac: float = 0.03
    concurrency: int = 50
    delay: float = 0.0
    long_pause_after: int = 3
    long_pause: float = 0.0
    batch: int = 1000
    seed: int = 0
    cleanup: bool = True
    out: str = "simulator/out/result.json"


def _extract_ids(body) -> list[str]:
    """Принимает оба формата ответа /recs."""
    if isinstance(body, dict):
        body = body.get("items") or body.get("recs") or body.get("recommendations") or []
    out = []
    for x in body:
        if isinstance(x, dict):
            x = x.get("item_id", x.get("id"))
        out.append(str(x))
    return out


async def _one_request(
    client: httpx.AsyncClient,
    cfg: Config,
    user: SimUser,
    genres: dict,
    stats: RoundStats,
    shown_all: set,
) -> None:
    started = time.perf_counter()
    resp = await client.get(f"{cfg.recs_url}/recs/{user.user_id}")
    stats.latency_ms.append((time.perf_counter() - started) * 1000)
    resp.raise_for_status()
    recs = _extract_ids(resp.json())
    if len(recs) < TOP_K:
        stats.short_lists += 1

    # Объект, показанный в прошлой сессии, в новой уже не релевантен.
    fresh_relevant = user.relevant - user.seen
    stats.precision.append(precision_at_k(recs, fresh_relevant, TOP_K))
    stats.ndcg.append(ndcg_at_k(recs, fresh_relevant, TOP_K))
    stats.diversity.append(diversity(recs, genres))
    shown_all.update(recs)

    for item in recs:
        payload = {"user_id": user.user_id, "item_id": item, "interaction": user.reaction(item)}
        await client.post(f"{cfg.events_url}/interact", json=payload)
    user.seen.update(recs)
    if cfg.delay:
        await asyncio.sleep(cfg.delay)


async def run(cfg: Config) -> dict:
    catalog = load_catalog(cfg.catalog_path, cfg.items, cfg.seed)
    genres = dict(catalog)
    users = make_users(catalog, cfg.users, cfg.relevant_frac, cfg.seed)
    n_genres = len({g for gs in genres.values() for g in gs})
    print(
        f"Каталог: {len(catalog)} объектов, {n_genres} жанров; "
        f"пользователей: {len(users)}, релевантных на пользователя: {len(users[0].relevant)}"
    )

    limits = httpx.Limits(max_connections=cfg.concurrency + 10)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        if cfg.cleanup:
            r = await client.post(f"{cfg.recs_url}/cleanup")
            r.raise_for_status()
        for start in range(0, len(catalog), cfg.batch):
            chunk = catalog[start : start + cfg.batch]
            r = await client.post(
                f"{cfg.recs_url}/add_items",
                json={"item_ids": [i for i, _ in chunk], "genres": [g for _, g in chunk]},
            )
            r.raise_for_status()

        sem = asyncio.Semaphore(cfg.concurrency)
        shown_all: set[str] = set()
        rounds: list[dict] = []
        total = RoundStats()
        t0 = time.perf_counter()

        async def guarded(user: SimUser, stats: RoundStats) -> None:
            async with sem:
                await _one_request(client, cfg, user, genres, stats, shown_all)

        for rnd in range(cfg.rounds):
            stats = RoundStats()
            await asyncio.gather(*(guarded(u, stats) for u in users))
            summary = stats.summary()
            summary["round"] = rnd + 1
            summary["coverage"] = len(shown_all) / len(catalog)
            rounds.append(summary)
            for name in ("precision", "ndcg", "diversity", "latency_ms"):
                getattr(total, name).extend(getattr(stats, name))
            total.short_lists += stats.short_lists
            print(
                f"раунд {rnd + 1}: precision={summary['precision']:.3f} ndcg={summary['ndcg']:.3f} "
                f"diversity={summary['diversity']:.3f} coverage={summary['coverage']:.3f} "
                f"p50={summary['latency_p50_ms']:.0f}мс p95={summary['latency_p95_ms']:.0f}мс"
            )
            if cfg.long_pause and rnd + 1 == cfg.long_pause_after:
                print(f"длинная пауза {cfg.long_pause:.0f} с (ждём переобучение pipeline)")
                await asyncio.sleep(cfg.long_pause)

        result = total.summary()
        result.update(
            coverage=len(shown_all) / len(catalog),
            rounds=rounds,
            config={k: v for k, v in cfg.__dict__.items()},
            wall_seconds=time.perf_counter() - t0,
        )
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
