"""Импорт приложений строит роуты FastAPI — ловим ошибки аннотаций без инфраструктуры."""

import importlib


def test_apps_import_and_expose_routes():
    recs = importlib.import_module("services.recommendations.main")
    events = importlib.import_module("services.event_collector.main")
    importlib.import_module("services.pipeline.main")
    recs_paths = {r.path for r in recs.app.routes}
    events_paths = {r.path for r in events.app.routes}
    assert {"/healthcheck", "/cleanup", "/add_items", "/recs/{user_id}", "/metrics"} <= recs_paths
    assert {"/healthcheck", "/interact", "/metrics"} <= events_paths
