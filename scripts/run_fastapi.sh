#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:."
uvicorn deployment.fastapi_server:app --host 0.0.0.0 --port 8000 --workers 1
