#!/bin/sh

# Inicia el servidor Uvicorn.
echo "Starting Uvicorn server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
