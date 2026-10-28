"""Нагрузочный тест /recs: N параллельных клиентов в течение T секунд.

python -m simulator.loadtest --seconds 20 --concurrency 50
Перед запуском каталог должен быть загружен (например, симулятором с --no-cleanup или вручную).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx

from simulator.metrics import percentile


async def worker(
    client: httpx.AsyncClient, url: str, deadline: float, lat: list[float], errors: list[int]
):
    while time.perf_counter() < deadline:
        user = f"load_{random.randrange(10_000)}"
        t = time.perf_counter()
        try:
            r = await client.get(f"{url}/recs/{user}")
            r.raise_for_status()
        except httpx.HTTPError:
            errors.append(1)
            continue
        lat.append((time.perf_counter() - t) * 1000)


async def run(url: str, seconds: float, concurrency: int) -> None:
    lat: list[float] = []
    errors: list[int] = []
    async with httpx.AsyncClient(
        timeout=30, limits=httpx.Limits(max_connections=concurrency + 5)
    ) as c:
        deadline = time.perf_counter() + seconds
        await asyncio.gather(*(worker(c, url, deadline, lat, errors) for _ in range(concurrency)))
    print(
        f"requests={len(lat)} rps={len(lat) / seconds:.0f} errors={len(errors)} "
        f"mean={sum(lat) / max(1, len(lat)):.1f}мс p50={percentile(lat, 0.5):.1f}мс "
        f"p95={percentile(lat, 0.95):.1f}мс p99={percentile(lat, 0.99):.1f}мс"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5001")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--concurrency", type=int, default=50)
    a = p.parse_args()
    asyncio.run(run(a.url, a.seconds, a.concurrency))
