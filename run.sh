#!/usr/bin/env bash
# Starts the NimbusMart backend (FastAPI) and frontend (Vite) together.
# Run from the project root:  ./run.sh
# Press Ctrl+C once to stop both.

set -euo pipefail

# Always work relative to this script's location, wherever it's called from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# When this script exits (Ctrl+C or error), kill the whole process group
# so neither the backend nor the frontend is left running in the background.
cleanup() {
  echo ""
  echo "Shutting down backend and frontend..."
  # Kill every process in this script's group; ignore "no such process".
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend on http://localhost:8000 ..."
(
  cd "$ROOT/backend"
  uv run uvicorn app.main:app --reload --port 8000
) &

echo "Starting frontend on http://localhost:5173 ..."
(
  cd "$ROOT/frontend"
  npm run dev
) &

echo ""
echo "Both are starting up. Open http://localhost:5173 in your browser."
echo "Press Ctrl+C to stop."

# Wait for both background jobs; if either exits, cleanup() tears down the rest.
wait
