#!/usr/bin/env bash
# Arranca la API (interna, puerto 8000) y la interfaz (pública, puerto 7860).
set -e
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run interfaz/app.py \
  --server.port 7860 --server.address 0.0.0.0 \
  --server.headless true --server.enableCORS false
