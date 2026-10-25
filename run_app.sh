#!/usr/bin/env bash
# ==============================================================================
# AI Image Detector - Web Application Runner
# Launches the FastAPI detection backend and React/Vite frontend.
# ==============================================================================

set -e

# Resolve repository root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Configuration defaults
HOST="127.0.0.1"
BACKEND_PORT=8000
FRONTEND_PORT=5173
MODE="both"  # "both" | "backend" | "frontend"

# Text styles
BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

print_banner() {
  echo -e "${CYAN}${BOLD}"
  echo "=========================================================="
  echo "         AI Image Verification Platform Runner            "
  echo "=========================================================="
  echo -e "${RESET}"
}

print_help() {
  echo "Usage: ./run_app.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --backend-only       Run only the FastAPI backend server"
  echo "  --frontend-only      Run only the Vite frontend dev server"
  echo "  --host <ip>          Host interface to bind to (default: 127.0.0.1)"
  echo "  --backend-port <p>   Port for FastAPI backend (default: 8000)"
  echo "  --frontend-port <p>  Port for Vite frontend (default: 5173)"
  echo "  -h, --help           Show this help message"
  echo ""
}

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-only)
      MODE="backend"
      shift
      ;;
    --frontend-only)
      MODE="frontend"
      shift
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --backend-port)
      BACKEND_PORT="$2"
      shift 2
      ;;
    --frontend-port)
      FRONTEND_PORT="$2"
      shift 2
      ;;
    -h|--help)
      print_banner
      print_help
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${RESET}"
      print_help
      exit 1
      ;;
  esac
done

print_banner

# Locate Python executable
PYTHON_BIN="python3"
if [[ -f "$PROJECT_ROOT/.venv/bin/python3" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="python3"
else
  echo -e "${RED}Error: Python 3 not found.${RESET}"
  exit 1
fi

# Locate Node / npm (including nvm auto-detection)
NPM_BIN=""
if command -v npm &>/dev/null; then
  NPM_BIN="npm"
elif command -v bun &>/dev/null; then
  NPM_BIN="bun"
else
  # Check common nvm and local paths
  NVM_NODE_BIN=$(ls -d "$HOME/.nvm/versions/node/"*/bin 2>/dev/null | tail -n 1)
  if [[ -n "$NVM_NODE_BIN" && -x "$NVM_NODE_BIN/npm" ]]; then
    export PATH="$NVM_NODE_BIN:$PATH"
    NPM_BIN="$NVM_NODE_BIN/npm"
  fi
fi

# Track child background PIDs for graceful termination
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo -e "${YELLOW}Shutting down application services...${RESET}"
  if [[ -n "$BACKEND_PID" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$FRONTEND_PID" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  echo -e "${GREEN}All services stopped cleanly.${RESET}"
}
trap cleanup SIGINT SIGTERM EXIT

# ------------------------------------------------------------------------------
# 1. Start Backend Server
# ------------------------------------------------------------------------------
start_backend() {
  echo -e "${CYAN}→ Starting FastAPI Detection Backend on http://${HOST}:${BACKEND_PORT}...${RESET}"
  export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

  "$PYTHON_BIN" -m aidetector.cli serve --host "$HOST" --port "$BACKEND_PORT" &
  BACKEND_PID=$!

  # Wait briefly for backend server to spin up
  echo -n "Waiting for backend health check..."
  for i in {1..20}; do
    if curl -s "http://${HOST}:${BACKEND_PORT}/api/health" &>/dev/null; then
      echo -e " ${GREEN}Ready!${RESET}"
      break
    fi
    sleep 0.5
    echo -n "."
  done
  echo ""
}

# ------------------------------------------------------------------------------
# 2. Start Frontend Server
# ------------------------------------------------------------------------------
start_frontend() {
  if [[ -z "$NPM_BIN" ]]; then
    echo -e "${YELLOW}Notice: Node.js/npm is not installed in the current environment.${RESET}"
    if [[ -d "$PROJECT_ROOT/dist" ]]; then
      echo -e "${GREEN}Pre-built frontend found in dist/. The backend will serve the web UI at:${RESET}"
      echo -e "  ${BOLD}${GREEN}http://${HOST}:${BACKEND_PORT}/${RESET}"
    else
      echo -e "${YELLOW}To run the React dev server, install Node.js (v18+) and run 'npm install && npm run dev'.${RESET}"
      echo -e "${CYAN}Backend API and Swagger docs remain available at:${RESET}"
      echo -e "  ${BOLD}${CYAN}http://${HOST}:${BACKEND_PORT}/docs${RESET}"
    fi
    return
  fi

  # Check node_modules
  if [[ ! -d "$PROJECT_ROOT/node_modules" ]]; then
    echo -e "${YELLOW}node_modules not found. Installing frontend dependencies...${RESET}"
    "$NPM_BIN" install
  fi

  echo -e "${CYAN}→ Starting Vite Frontend Dev Server on http://localhost:${FRONTEND_PORT}...${RESET}"
  "$NPM_BIN" run dev -- --port "$FRONTEND_PORT" &
  FRONTEND_PID=$!
}

# ------------------------------------------------------------------------------
# Execution based on selected mode
# ------------------------------------------------------------------------------
if [[ "$MODE" == "backend" ]]; then
  start_backend
  echo -e "${GREEN}${BOLD}Backend is running:${RESET}"
  echo -e "  API URL:  ${CYAN}http://${HOST}:${BACKEND_PORT}${RESET}"
  echo -e "  API Docs: ${CYAN}http://${HOST}:${BACKEND_PORT}/docs${RESET}"
  echo "Press Ctrl+C to terminate."
  wait "$BACKEND_PID"

elif [[ "$MODE" == "frontend" ]]; then
  start_frontend
  if [[ -n "$FRONTEND_PID" ]]; then
    echo -e "${GREEN}${BOLD}Frontend is running:${RESET}"
    echo -e "  Web App:  ${CYAN}http://localhost:${FRONTEND_PORT}${RESET}"
    echo "Press Ctrl+C to terminate."
    wait "$FRONTEND_PID"
  fi

else # "both"
  start_backend
  start_frontend

  echo -e "${GREEN}${BOLD}==========================================================${RESET}"
  echo -e "${GREEN}${BOLD}  Web App Services Running Successfully!                 ${RESET}"
  echo -e "${GREEN}${BOLD}==========================================================${RESET}"
  if [[ -n "$FRONTEND_PID" ]]; then
    echo -e "  Frontend UI:  ${BOLD}${GREEN}http://localhost:${FRONTEND_PORT}${RESET}"
  else
    echo -e "  Backend UI:   ${BOLD}${GREEN}http://${HOST}:${BACKEND_PORT}${RESET}"
  fi
  echo -e "  Backend API:  ${CYAN}http://${HOST}:${BACKEND_PORT}${RESET}"
  echo -e "  Swagger Docs: ${CYAN}http://${HOST}:${BACKEND_PORT}/docs${RESET}"
  echo ""
  echo "Press Ctrl+C to stop all services."

  # Wait on either process
  if [[ -n "$FRONTEND_PID" ]]; then
    wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || wait "$BACKEND_PID"
  else
    wait "$BACKEND_PID"
  fi
fi
