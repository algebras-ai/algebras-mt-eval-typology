#!/usr/bin/env bash
# Option B: refresh gh token scopes, create org repo if missing, then push.
# Avoids `gh repo create --remote=origin` when origin is already configured locally.
set -euo pipefail
cd "$(dirname "$0")/.."

ORG_REPO="algebras-ai/algebras-mt-eval-typology"
HTTPS_URL="https://github.com/${ORG_REPO}.git"
SSH_URL="git@github.com:${ORG_REPO}.git"

echo "=== 1) Refresh token (if needed) ==="
gh auth refresh -h github.com -s repo,write:org,read:org || true

echo "=== 2) Check remote repository ==="
if gh repo view "${ORG_REPO}" &>/dev/null; then
  echo "Repository ${ORG_REPO} already exists on GitHub."
else
  echo "Creating empty repository on GitHub (without replacing origin)..."
  gh repo create "${ORG_REPO}" \
    --public \
    --description "Reproduce paper: typological distance vs LLM judge vs chrF (WMT25)"
fi

echo "=== 3) Point origin and push ==="
if git remote get-url origin &>/dev/null; then
  # Keep SSH if current URL is ssh, otherwise HTTPS
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

echo "Done: https://github.com/${ORG_REPO}"
