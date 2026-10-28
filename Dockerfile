FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Матрицы крошечные: многопоточный BLAS только мешает (потоки разных воркеров конкурируют за CPU).
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY src ./src
COPY services ./services

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 5000 5001 8000

# По умолчанию — сервис рекомендаций; остальные сервисы переопределяют command в compose.
CMD ["uvicorn", "services.recommendations.main:app", "--host", "0.0.0.0", "--port", "5001"]
