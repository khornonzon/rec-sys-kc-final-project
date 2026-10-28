.PHONY: help venv lint fmt test up down build logs ps simulate deploy clean

PY ?= .venv/bin/python
VM ?=
VM_USER ?= root
VM_DIR ?= /opt/recsys-service

help:            ## Список команд
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:            ## Локальное окружение для тестов и симулятора
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

lint:            ## ruff check + format --check
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

fmt:             ## Автоформатирование
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .

test:            ## Unit-тесты (без инфраструктуры)
	$(PY) -m pytest -q

test-e2e:        ## Интеграционные тесты против поднятого docker compose
	$(PY) -m pytest -q -m integration

build:           ## Сборка образа
	docker compose build

up:              ## Поднять весь стек
	docker compose up -d --build

down:            ## Остановить стек и удалить тома
	docker compose down -v

logs:            ## Логи сервисов
	docker compose logs -f --tail=100 recommendations event-collector pipeline

ps:              ## Состояние контейнеров
	docker compose ps

simulate:        ## Прогон симулятора грейдера против localhost
	$(PY) -m simulator --users 1000 --rounds 3

deploy:          ## Копировать проект на ВМ и запустить: make deploy VM=1.2.3.4
	@test -n "$(VM)" || (echo "Укажите VM=<ip>"; exit 1)
	rsync -az --delete --exclude .venv --exclude .git --exclude simulator/out ./ $(VM_USER)@$(VM):$(VM_DIR)/
	ssh $(VM_USER)@$(VM) "cd $(VM_DIR) && bash deploy/setup_vm.sh"

clean:           ## Удалить кэши
	rm -rf .pytest_cache .ruff_cache simulator/out
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
