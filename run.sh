#!/usr/bin/env bash
# Telecom Customer Churn Prediction - Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found! Please ensure .venv exists."
    exit 1
fi

echo "Starting Telecom Customer Churn Prediction Web App..."
exec .venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
