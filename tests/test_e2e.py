"""Интеграционный сценарий против поднятого docker compose.

Запуск: make test-e2e (или pytest -m integration). Адреса берутся из RECS_URL / EVENTS_URL.
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

RECS_URL = os.getenv("RECS_URL", "http://localhost:5001")
EVENTS_URL = os.getenv("EVENTS_URL", "http://localhost:5000")

CATALOG = {
    "item_ids": [1, 2, 3, "tt4", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "genres": [
        ["Drama"], ["Comedy"], ["Drama", "Romance"], ["Action"], ["Horror"],
        ["Sci-Fi"], ["Comedy", "Romance"], ["Action", "Thriller"], ["Drama"],
        ["Animation"], ["Horror", "Thriller"], ["Documentary"], ["Horror"],
        ["Horror", "Mystery"], ["Comedy"],
    ],
}  # fmt: skip


def _ids(body):
    return body["items"] if isinstance(body, dict) else body


def test_full_grader_protocol():
    with httpx.Client(timeout=10) as c:
        assert c.get(f"{RECS_URL}/healthcheck").status_code == 200
        assert c.get(f"{EVENTS_URL}/healthcheck").status_code == 200
        assert c.post(f"{RECS_URL}/cleanup").status_code == 200
        assert c.post(f"{RECS_URL}/add_items", json=CATALOG).json()["added"] == 15

        first = _ids(c.get(f"{RECS_URL}/recs/e2e_user").json())
        assert len(first) == 10 == len(set(first))
        assert set(first) <= set(CATALOG["item_ids"])  # типы id сохранены (int и str)

        for item in first:
            reaction = "like" if item in (5, 11, 13, 14) else "dislike"
            r = c.post(
                f"{EVENTS_URL}/interact",
                json={"user_id": "e2e_user", "item_id": item, "interaction": reaction},
            )
            assert r.json()["status"] == "ok"

        second = _ids(c.get(f"{RECS_URL}/recs/e2e_user").json())
        assert not set(first) & set(second)  # показанное не повторяется
        assert len(first) + len(second) == 15  # каталог исчерпан за два запроса
        assert _ids(c.get(f"{RECS_URL}/recs/e2e_user").json()) == []

        assert c.post(f"{RECS_URL}/cleanup").status_code == 200
        assert _ids(c.get(f"{RECS_URL}/recs/e2e_user").json()) == []  # каталог пуст после сброса
