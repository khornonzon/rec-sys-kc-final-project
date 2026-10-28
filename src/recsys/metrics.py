"""Prometheus-метрики. Экспортируются на /metrics каждого сервиса."""

from prometheus_client import Counter, Gauge, Histogram

RECS_LATENCY = Histogram(
    "recs_request_seconds",
    "Время формирования рекомендаций",
    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.8, 2.0),
)
RECS_RETURNED = Histogram(
    "recs_items_returned", "Сколько объектов вернули", buckets=(0, 1, 5, 9, 10, 20)
)
RECS_CF_USED = Counter("recs_cf_used_total", "Запросов, где использовались ALS-векторы")
RECS_PROFILE_USED = Counter("recs_profile_used_total", "Запросов с непустым контентным профилем")
INTERACTIONS = Counter("interactions_total", "Принятые события фидбека", ["reaction"])
INTERACTIONS_REJECTED = Counter("interactions_rejected_total", "Нераспознанные тела /interact")
PUBLISH_ERRORS = Counter("rabbit_publish_errors_total", "Ошибки публикации в RabbitMQ")
CATALOG_SIZE = Gauge("catalog_items", "Размер каталога в снапшоте сервиса")
SNAPSHOT_AGE = Gauge("snapshot_age_seconds", "Возраст снапшота счётчиков")
PIPELINE_EVENTS = Counter("pipeline_events_total", "События, прочитанные из очереди")
PIPELINE_TRAININGS = Counter("pipeline_trainings_total", "Успешные обучения ALS")
PIPELINE_TRAIN_SECONDS = Histogram("pipeline_train_seconds", "Длительность обучения ALS")
PIPELINE_LAST_LIKES = Gauge("pipeline_train_likes", "Лайков в последней обучающей матрице")
