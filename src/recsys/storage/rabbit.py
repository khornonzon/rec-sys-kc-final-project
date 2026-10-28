"""RabbitMQ: публикация событий фидбека и их потребление в pipeline."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import aio_pika
import orjson
from aio_pika.abc import AbstractRobustConnection

log = logging.getLogger(__name__)


class RabbitPublisher:
    def __init__(self, url: str, queue: str) -> None:
        self.url = url
        self.queue = queue
        self._conn: AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self) -> None:
        self._conn = await aio_pika.connect_robust(self.url)
        self._channel = await self._conn.channel()
        await self._channel.declare_queue(self.queue, durable=True)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed

    async def publish(self, payload: dict) -> None:
        if self._channel is None:
            raise RuntimeError("publisher не подключён")
        msg = aio_pika.Message(
            body=orjson.dumps(payload),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await self._channel.default_exchange.publish(msg, routing_key=self.queue)

    async def purge(self) -> None:
        if self._channel is None:
            return
        queue = await self._channel.declare_queue(self.queue, durable=True)
        await queue.purge()


async def consume(url: str, queue_name: str, prefetch: int = 100) -> AsyncIterator[dict]:
    """Бесконечный итератор событий из очереди с подтверждением после обработки."""
    conn = await aio_pika.connect_robust(url)
    async with conn:
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=prefetch)
        queue = await channel.declare_queue(queue_name, durable=True)
        async with queue.iterator() as it:
            async for message in it:
                async with message.process(ignore_processed=True):
                    try:
                        yield orjson.loads(message.body)
                    except orjson.JSONDecodeError:
                        log.warning("Пропущено нечитаемое сообщение: %r", message.body[:200])
