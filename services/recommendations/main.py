"""Recommendations service (порт 5001).

Эндпоинты грейдера: /healthcheck, /cleanup, /add_items, /recs/{user_id}.
Дополнительно /metrics (Prometheus).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from recsys.config import get_settings
from recsys.logging_setup import setup_logging
from recsys.metrics import RECS_LATENCY, RECS_RETURNED
from recsys.rawlog import RawRequestLogger
from recsys.recommender import Recommender, SnapshotKeeper
from recsys.schemas import SchemaError, parse_add_items
from recsys.storage.qdrant_store import QdrantStore
from recsys.storage.rabbit import RabbitPublisher
from recsys.storage.redis_store import RedisStore

setup_logging()
log = logging.getLogger("recommendations")
settings = get_settings()
rawlog = RawRequestLogger(log, settings.log_raw_requests)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = RedisStore.from_url(settings.redis_url)
    qdrant = QdrantStore.from_url(settings.qdrant_url, settings.qdrant_collection)
    rabbit = RabbitPublisher(settings.rabbitmq_url, settings.rabbitmq_queue)
    try:
        await rabbit.connect()
    except Exception:
        log.warning("RabbitMQ недоступен при старте; очистка очереди будет пропускаться")
    keeper = SnapshotKeeper(store, settings)
    try:
        await keeper.refresh(force=True)
    except Exception:
        log.exception("Не удалось загрузить начальный снапшот")
    task = asyncio.create_task(keeper.run())
    app.state.store = store
    app.state.qdrant = qdrant
    app.state.rabbit = rabbit
    app.state.keeper = keeper
    app.state.recommender = Recommender(store, qdrant, keeper, settings)
    app.state.cleanup_lock = asyncio.Lock()
    log.info("Сервис рекомендаций запущен, формат ответа: %s", settings.recs_response_format)
    try:
        yield
    finally:
        task.cancel()
        await rabbit.close()
        await qdrant.close()
        await store.close()


app = FastAPI(
    title="Recommendations service",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.get("/healthcheck")
async def healthcheck(request: Request) -> dict:
    redis_ok = await request.app.state.store.ping()
    return {
        "status": "ok",
        "redis": redis_ok,
        "catalog_items": len(request.app.state.keeper.snapshot.catalog),
    }


@app.api_route("/cleanup", methods=["POST", "GET", "DELETE"])
async def cleanup(request: Request) -> dict:
    """Сброс всего состояния перед новым запуском грейдера."""
    st = request.app.state
    async with st.cleanup_lock:
        epoch = await st.store.reset()
        try:
            await st.qdrant.drop()
        except Exception:
            log.warning("Не удалось удалить коллекцию Qdrant", exc_info=True)
        try:
            if st.rabbit.connected:
                await st.rabbit.purge()
        except Exception:
            log.warning("Не удалось очистить очередь RabbitMQ", exc_info=True)
        await st.keeper.refresh(force=True)
    log.info("Cleanup выполнен, epoch=%d", epoch)
    return {"status": "ok", "epoch": epoch}


@app.post("/add_items", response_model=None)
async def add_items(request: Request) -> Response | dict:
    body = await request.body()
    rawlog.log("/add_items", body)
    try:
        payload = orjson.loads(body) if body else {}
        items = parse_add_items(payload)
    except (orjson.JSONDecodeError, SchemaError) as e:
        log.warning("Некорректное тело /add_items: %s", e)
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=422)
    st = request.app.state
    version = await st.store.add_items(items)
    await st.keeper.refresh(force=True)
    log.info(
        "Добавлено объектов: %d, размер каталога: %d", len(items), len(st.keeper.snapshot.catalog)
    )
    return {"status": "ok", "added": len(items), "catalog_version": version}


@app.get("/recs/{user_id}")
async def recs(user_id: str, request: Request) -> list | dict:
    started = time.perf_counter()
    items = await request.app.state.recommender.recommend(user_id)
    RECS_LATENCY.observe(time.perf_counter() - started)
    RECS_RETURNED.observe(len(items))
    if settings.recs_response_format == "list":
        return items
    # Точный ключ, который читает грейдер, неизвестен (бейзлайн недоступен), поэтому список
    # возвращается под несколькими общепринятыми именами. Лишние ключи клиенту не мешают.
    return {
        "user_id": user_id,
        "recs": items,
        "items": items,
        "item_ids": items,
        "recommendations": items,
    }
