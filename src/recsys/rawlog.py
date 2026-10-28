"""Логирование первых N сырых тел запросов — чтобы увидеть реальный формат грейдера."""

import itertools
import logging


class RawRequestLogger:
    def __init__(self, logger: logging.Logger, limit: int) -> None:
        self.logger = logger
        self.limit = limit
        self._counter = itertools.count()

    def log(self, route: str, body: bytes) -> None:
        n = next(self._counter)
        if n < self.limit:
            self.logger.info("RAW %s #%d: %s", route, n, body[:2000].decode("utf-8", "replace"))
