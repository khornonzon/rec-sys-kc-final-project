"""Схемы запросов грейдера.

Точный формат запросов неизвестен (бейзлайн недоступен), поэтому парсинг
намеренно терпимый: принимаем несколько вариантов имён полей и значений.
Все распознанные варианты перечислены в константах ниже, чтобы их было легко
расширить, увидев реальные тела запросов в логах.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from recsys.ids import canonical, to_json_literal

Reaction = Literal["like", "dislike"]

USER_KEYS = ("user_id", "user", "uid", "userId")
ITEM_KEYS = ("item_id", "item", "iid", "itemId", "object_id", "movie_id", "id")
REACTION_KEYS = (
    "interaction",
    "reaction",
    "event",
    "event_type",
    "type",
    "action",
    "feedback",
    "rating",
    "value",
    "liked",
    "is_like",
)
# Параллельные списки: {"item_ids": [...], "actions": [...]} — формат грейдера
LIST_REACTION_KEYS = ("actions", "interactions", "reactions", "events", "feedbacks", "types")
LIKE_VALUES = {"like", "likes", "liked", "1", "true", "+1", "positive", "up"}
DISLIKE_VALUES = {"dislike", "dislikes", "disliked", "-1", "0", "false", "negative", "down"}


@dataclass(frozen=True, slots=True)
class Item:
    key: str
    orig: str
    genres: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Interaction:
    user: str
    item: str
    reaction: Reaction


class SchemaError(ValueError):
    """Тело запроса не удалось интерпретировать."""


def _normalize_genres(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw.strip(),) if raw.strip() else ()
    if isinstance(raw, list | tuple | set):
        out: list[str] = []
        for g in raw:
            if g is None:
                continue
            s = str(g).strip()
            if s and s not in out:
                out.append(s)
        return tuple(out)
    return (str(raw),)


def parse_add_items(payload: Any) -> list[Item]:
    """Разбор тела /add_items.

    Основной формат из README: {"item_ids": [...], "genres": [[...], ...]}.
    Дополнительно принимаем {"items": [{"item_id": ..., "genres": [...]}, ...]}
    и просто список таких объектов.
    """
    if isinstance(payload, dict) and "item_ids" in payload:
        ids = payload["item_ids"]
        genres = payload.get("genres") or []
        if not isinstance(ids, list):
            raise SchemaError("item_ids должен быть списком")
        if not isinstance(genres, list):
            raise SchemaError("genres должен быть списком списков")
        if genres and len(genres) != len(ids):
            raise SchemaError("длины item_ids и genres не совпадают")
        return [
            Item(canonical(i), to_json_literal(i), _normalize_genres(genres[n] if genres else None))
            for n, i in enumerate(ids)
        ]

    records: Any = None
    if isinstance(payload, dict):
        for k in ("items", "objects", "data"):
            if isinstance(payload.get(k), list):
                records = payload[k]
                break
    elif isinstance(payload, list):
        records = payload
    if records is None:
        raise SchemaError("не найдено ни item_ids, ни списка объектов")

    items: list[Item] = []
    for rec in records:
        if not isinstance(rec, dict):
            raise SchemaError("элемент списка объектов должен быть словарём")
        item_id = next((rec[k] for k in ITEM_KEYS if k in rec), None)
        if item_id is None:
            raise SchemaError("у объекта нет идентификатора")
        g = rec.get("genres", rec.get("genre"))
        items.append(Item(canonical(item_id), to_json_literal(item_id), _normalize_genres(g)))
    return items


def _parse_reaction(value: Any) -> Reaction | None:
    if isinstance(value, bool):
        return "like" if value else "dislike"
    if isinstance(value, int | float):
        return "like" if value > 0 else "dislike"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in LIKE_VALUES:
            return "like"
        if v in DISLIKE_VALUES:
            return "dislike"
    return None


def _parse_one(rec: dict[str, Any], default_user: Any = None) -> list[Interaction]:
    user = next((rec[k] for k in USER_KEYS if k in rec), default_user)
    if user is None:
        raise SchemaError("не найден user_id")

    # Формат грейдера: списки item_ids и actions одной длины (или одно действие на все объекты).
    items_list = rec.get("item_ids")
    if isinstance(items_list, list):
        for k in LIST_REACTION_KEYS:
            if isinstance(rec.get(k), list):
                actions = rec[k]
                if len(actions) == 1 and len(items_list) > 1:
                    actions = actions * len(items_list)
                if len(actions) != len(items_list):
                    raise SchemaError(f"длины item_ids и {k} не совпадают")
                out: list[Interaction] = []
                for item, act in zip(items_list, actions, strict=True):
                    reaction = _parse_reaction(act)
                    if reaction is None:
                        raise SchemaError(f"неизвестное действие: {act!r}")
                    out.append(Interaction(canonical(user), canonical(item), reaction))
                return out

    reaction: Reaction | None = None
    for k in REACTION_KEYS:
        if k in rec:
            reaction = _parse_reaction(rec[k])
            if reaction is not None:
                break
    # Вариант {"like": true} / {"dislike": true}
    if reaction is None:
        if rec.get("like") is True:
            reaction = "like"
        elif rec.get("dislike") is True:
            reaction = "dislike"
    if reaction is None:
        raise SchemaError("не удалось распознать тип реакции")

    item = next((rec[k] for k in ITEM_KEYS if k in rec), None)
    if item is not None:
        return [Interaction(canonical(user), canonical(item), reaction)]
    if isinstance(rec.get("item_ids"), list):
        return [Interaction(canonical(user), canonical(i), reaction) for i in rec["item_ids"]]
    raise SchemaError("не найден item_id")


def parse_interactions(payload: Any) -> list[Interaction]:
    """Разбор тела /interact: одиночное событие, список событий или обёртка со списком."""
    if isinstance(payload, list):
        out: list[Interaction] = []
        for rec in payload:
            if not isinstance(rec, dict):
                raise SchemaError("событие должно быть словарём")
            out.extend(_parse_one(rec))
        return out
    if not isinstance(payload, dict):
        raise SchemaError("тело запроса должно быть объектом или списком")

    for k in ("events", "interactions", "items", "data"):
        if isinstance(payload.get(k), list):
            default_user = next((payload[u] for u in USER_KEYS if u in payload), None)
            out = []
            for rec in payload[k]:
                if not isinstance(rec, dict):
                    raise SchemaError("событие должно быть словарём")
                out.extend(_parse_one(rec, default_user))
            return out
    return _parse_one(payload)
