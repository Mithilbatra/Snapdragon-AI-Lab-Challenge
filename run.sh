#!/usr/bin/env bash
# Start SnapAssist locally (macOS / Linux, CPU dev backend).
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q -r requirements-dev.txt
fi

exec ./.venv/bin/python -m app.server
