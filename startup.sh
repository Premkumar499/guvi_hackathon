#!/bin/bash

# Memory optimization environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536

# Download model if needed
python download_model.py

# Start gunicorn with memory-optimized settings
exec gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --timeout 300 \
    --workers 1 \
    --threads 1 \
    --worker-class sync \
    --max-requests 50 \
    --max-requests-jitter 5 \
    --preload
