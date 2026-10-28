"""Настройки всех сервисов. Читаются из переменных окружения и файла .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Инфраструктура
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "items_cf"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_queue: str = "interactions"

    # API
    recs_response_format: Literal["dict", "list"] = "dict"
    top_k: int = 10
    log_raw_requests: int = 20

    # Гибридный скоринг
    w_content: float = 0.4
    w_cf: float = 0.3
    w_pop: float = 1.0
    w_explore: float = 0.1
    dislike_weight: float = 0.5
    mmr_lambda: float = 0.9
    n_explore: int = 1
    pop_mode: Literal["ctr", "legacy"] = "ctr"
    pop_sampling: Literal["mean", "thompson"] = "mean"
    pop_jitter: float = 0.1
    ctr_prior_strength: float = 10.0
    genre_prior_smoothing: float = 50.0
    n_candidates: int = 200
    cf_candidates: int = 100
    snapshot_refresh_sec: float = 5.0

    # Pipeline (ALS)
    train_interval_sec: float = 60.0
    min_likes_to_train: int = 50
    als_factors: int = 32
    als_iterations: int = 10
    als_reg: float = 0.1
    als_alpha: float = 10.0
    cf_top_n: int = 50
    pipeline_metrics_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
