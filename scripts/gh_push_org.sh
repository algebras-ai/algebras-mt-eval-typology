#!/usr/bin/env bash
# Вариант B: обновить scopes gh, при необходимости создать репо в org, затем push.
# Не используем `gh repo create --remote=origin`, если origin уже настроен локально.
set -euo pipefail
cd "$(dirname "$0")/.."

ORG_REPO="algebras-ai/algebras-mt-eval-typology"
HTTPS_URL="https://github.com/${ORG_REPO}.git"
SSH_URL="git@github.com:${ORG_REPO}.git"

echo "=== 1) Обновление токена (при необходимости) ==="
gh auth refresh -h github.com -s repo,write:org,read:org || true

echo "=== 2) Проверка удалённого репозитория ==="
if gh repo view "${ORG_REPO}" &>/dev/null; then
  echo "Репозиторий ${ORG_REPO} уже есть на GitHub."
else
  echo "Создаём пустой репозиторий на GitHub (без подмены origin)..."
  gh repo create "${ORG_REPO}" \
    --public \
    --description "Reproduce paper: typological distance vs LLM judge vs chrF (WMT25)"
fi

echo "=== 3) Привязка origin и push ==="
if git remote get-url origin &>/dev/null; then
  # Сохраняем предпочтение: SSH, если текущий URL ssh, иначе HTTPS
  cur=$(git remote get-url origin)
  if [[ "$cur" == git@* ]]; then
    git remote set-url origin "${SSH_URL}"
  else
    git remote set-url origin "${HTTPS_URL}"
  fi
else
  git remote add origin "${SSH_URL}"
fi

git push -u origin main

echo "Готово: https://github.com/${ORG_REPO}"
