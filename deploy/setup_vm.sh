#!/usr/bin/env bash
# Разворачивает проект на свежей Ubuntu-ВМ: docker + compose plugin, затем docker compose up.
# Запускать на ВМ из корня репозитория: bash deploy/setup_vm.sh (sudo подхватится сам)
set -euo pipefail

# Нужны права root (apt, docker). Если запущено обычным пользователем — перезапускаемся через sudo.
if [ "$(id -u)" -ne 0 ]; then
  exec sudo -E bash "$0" "$@"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo ">>> Устанавливаю Docker"
  apt-get update -y
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
fi

if ! docker compose version >/dev/null 2>&1; then
  echo ">>> Устанавливаю docker compose plugin"
  apt-get install -y docker-compose-plugin
fi

cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env

echo ">>> Собираю и запускаю сервисы"
docker compose up -d --build
docker compose ps

echo ">>> Проверка здоровья"
sleep 5
curl -fsS http://localhost:5001/healthcheck && echo
curl -fsS http://localhost:5000/healthcheck && echo
echo "Готово. Порты 5000 (event collector) и 5001 (recommendations) открыты на всех интерфейсах."
