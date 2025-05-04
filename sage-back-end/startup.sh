#!/bin/bash

# Activate virtual environment
source env/scripts/activate

# Set environment variables to control memory usage
export PYTHONUNBUFFERED=1
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536

# Start with gunicorn using optimized config
gunicorn --config gunicorn_config.py server:app