"""Канонизация идентификаторов.

Грейдер может присылать item_id как число или как строку. Внутри системы
используем строковый ключ, а исходное JSON-представление сохраняем отдельно,
чтобы вернуть идентификатор ровно в том виде, в котором он пришёл.
"""

from typing import Any

import orjson


def canonical(value: Any) -> str:
    """Строковый ключ для Redis/Qdrant: строки как есть, остальное через JSON."""
    if isinstance(value, str):
        return value
    return orjson.dumps(value).decode()


def to_json_literal(value: Any) -> str:
    """Исходное представление значения в виде JSON-литерала."""
    return orjson.dumps(value).decode()


def from_json_literal(literal: str) -> Any:
    """Обратное преобразование литерала в исходное значение."""
    return orjson.loads(literal)
