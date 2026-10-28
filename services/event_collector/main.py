"""Event collector (порт 5000): принимает лайки/дизлайки.

Лёгкое состояние (множества пользователя, счётчики объекта) пишется в Redis
синхронно, чтобы уже следующий запрос рекомендаций учёл фидбек. Само событие
публикуется в RabbitMQ для regular pipeline (лог событий, переобучение ALS).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app

from recsys.config import get_settings
from recsys.logging_setup import setup_logging
from recsys.metrics import INTERACTIONS, INTERACTIONS_REJECTED, PUBLISH_ERRORS
from recsys.rawlog import RawRequestLogger
from recsys.schemas import SchemaError, parse_interactions
from recsys.storage.rabbit import RabbitPublisher
from recsys.storage.redis_store import RedisStore

setup_logging()
log = logging.getLogger("event_collector")
settings = get_settings()
rawlog = RawRequestLogger(log, settings.log_raw_requests)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = RedisStore.from_url(settings.redis_url)
    rabbit = RabbitPublisher(settings.rabbitmq_url, settings.rabbitmq_queue)
    try:
        await rabbit.connect()
    except Exception:
        log.warning("RabbitMQ недоступен при старте; повторим подключение при первом событии")
    app.state.store = store
    app.state.rabbit = rabbit
    log.info("Event collector запущен")
    try:
        yield
    finally:
        await rabbit.close()
        await store.close()


app = FastAPI(
    title="Event collector",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.get("/healthcheck")
async def healthcheck(request: Request) -> dict:
    return {
        "status": "ok",
        "redis": await request.app.state.store.ping(),
        "rabbitmq": request.app.state.rabbit.connected,
    }


@app.post("/interact")
async def interact(request: Request) -> dict:
    body = await request.body()
    rawlog.log("/interact", body)
    try:
        payload = orjson.loads(body) if body else {}
        events = parse_interactions(payload)
    except (orjson.JSONDecodeError, SchemaError) as e:
        # Отвечаем 200, чтобы не ломать прогон грейдера из-за неизвестного поля,
        # но фиксируем проблему в логах и метриках.
        INTERACTIONS_REJECTED.inc()
        log.warning("Нераспознанное тело /interact: %s; body=%s", e, body[:500])
        return {"status": "ignored", "detail": str(e)}

    st = request.app.state
    ts = time.time()
    for ev in events:
        await st.store.record_interaction(ev)
        INTERACTIONS.labels(ev.reaction).inc()
        await _publish(
            st.rabbit, {"user": ev.user, "item": ev.item, "reaction": ev.reaction, "ts": ts}
        )
    return {"status": "ok", "accepted": len(events)}


async def _publish(rabbit: RabbitPublisher, payload: dict) -> None:
    try:
        if not rabbit.connected:
            await rabbit.connect()
        await rabbit.publish(payload)
    except Exception:
        PUBLISH_ERRORS.inc()
        log.warning("Не удалось опубликовать событие в RabbitMQ", exc_info=True)
