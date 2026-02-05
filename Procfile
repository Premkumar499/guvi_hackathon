web: python download_model.py && gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 1 --worker-class sync --max-requests 50 --max-requests-jitter 5 --preload
