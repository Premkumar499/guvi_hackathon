#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# Download model if it doesn't exist
if [ ! -f "models/deepfake_model_v2.pth" ]; then
    echo "📥 Model not found, downloading..."
    python download_model.py
else
    echo "✅ Model already exists"
fi

# Start the application
echo "🔥 Starting gunicorn..."
exec gunicorn app:app
