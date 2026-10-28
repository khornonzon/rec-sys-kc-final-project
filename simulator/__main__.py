"""CLI: python -m simulator --users 300 --rounds 6 [--catalog path/to/imdb_movies.parquet]"""

from __future__ import annotations

import argparse
import asyncio

from simulator.runner import Config, run


def main() -> None:
    p = argparse.ArgumentParser(description="Эмулятор грейдера для сервиса рекомендаций")
    d = Config()
    p.add_argument("--recs-url", default=d.recs_url)
    p.add_argument("--events-url", default=d.events_url)
    p.add_argument("--users", type=int, default=d.users)
    p.add_argument("--rounds", type=int, default=d.rounds)
    p.add_argument("--items", type=int, default=d.items)
    p.add_argument(
        "--catalog",
        dest="catalog_path",
        default=d.catalog_path,
        help="parquet с колонками title_id, genre (IMDB из урока 2)",
    )
    p.add_argument("--relevant-frac", type=float, default=d.relevant_frac)
    p.add_argument("--concurrency", type=int, default=d.concurrency)
    p.add_argument(
        "--delay", type=float, default=d.delay, help="пауза после запроса пользователя, с"
    )
    p.add_argument("--long-pause-after", type=int, default=d.long_pause_after)
    p.add_argument(
        "--long-pause",
        type=float,
        default=d.long_pause,
        help="длинная пауза после указанного раунда, с (дать pipeline обучиться)",
    )
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--no-cleanup", dest="cleanup", action="store_false")
    p.add_argument("--out", default=d.out)
    args = p.parse_args()
    cfg = Config(**vars(args))
    result = asyncio.run(run(cfg))
    print("\nИтого:")
    for k in (
        "precision",
        "ndcg",
        "diversity",
        "coverage",
        "latency_p50_ms",
        "latency_p95_ms",
        "short_lists",
    ):
        v = result[k]
        print(f"  {k:16s} {v:.3f}" if isinstance(v, float) else f"  {k:16s} {v}")
    print(f"  результат сохранён в {cfg.out}")


if __name__ == "__main__":
    main()
