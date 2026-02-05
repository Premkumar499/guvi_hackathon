MODEL_PATH = "models/deepfake_model_v2.pth"

SAMPLE_RATE = 16000
MAX_LEN = SAMPLE_RATE * 2   # 2 seconds (reduced for memory optimization)
THRESHOLD = 0.45

# Always use CPU for memory optimization
DEVICE = "cpu"
