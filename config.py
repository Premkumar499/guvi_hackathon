MODEL_PATH = "models/deepfake_model_v2.pth"

SAMPLE_RATE = 16000
MAX_LEN = SAMPLE_RATE * 3   # 3 seconds
THRESHOLD = 0.45

# Device will be automatically set to "cpu" if CUDA is not available
DEVICE = "cuda"
