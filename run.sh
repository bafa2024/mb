#!/bin/bash
# Development run script

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run the application
uvicorn app:app --reload --host 0.0.0.0 --port 8000