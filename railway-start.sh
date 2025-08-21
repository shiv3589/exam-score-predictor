#!/bin/bash
echo "Railway Streamlit Startup Script"
echo "PORT environment variable: $PORT"
echo "Starting Streamlit on 0.0.0.0:$PORT"

exec streamlit run app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.enableCORS false \
  --server.enableXsrfProtection false
