"""Логирование первых N сырых тел запросов — чтобы увидеть реальный формат грейдера."""

import logging
from collections import Counter


class RawRequestLogger:
    def __init__(self, logger: logging.Logger, limit: int) -> None:
        self.logger = logger
        self.limit = limit
        self._counter: Counter[str] = Counter()

    def log(self, route: str, body: bytes) -> None:
        """Первые `limit` тел для каждого маршрута отдельно."""
        n = self._counter[route]
        self._counter[route] += 1
        if n < self.limit:
            self.logger.info("RAW %s #%d: %s", route, n, body[:2000].decode("utf-8", "replace"))
