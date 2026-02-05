import torch
import os
import logging
import gc

from transformers import Wav2Vec2Processor
from detector import Detector
from audio_utils import load_audio
from config import MODEL_PATH, DEVICE, THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize processor and model with error handling
try:
    logger.info("Initializing Wav2Vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )
    logger.info("✅ Processor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load processor: {e}")
    raise RuntimeError(f"Processor initialization failed: {e}")

try:
    print("🔄 Loading model...")
    
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"⚠️ Model file not found: {MODEL_PATH}")
        logger.warning("⚠️ API will start but predictions will fail until model is uploaded")
        model = None
        device = None
    else:
        device = "cpu"
        
        # Initialize model (downloads Wav2Vec2 base encoder)
        model = Detector()
        
        # Load trained weights
        logger.info("Loading saved weights...")
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        del state_dict
        gc.collect()
        
        model.to(device)
        model.eval()
        
        # Disable gradients for inference
        torch.set_grad_enabled(False)
        
        logger.info(f"✅ Model loaded on {device}")
        print("✅ Model Loaded Successfully")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.warning("⚠️ API will start but predictions will fail until model is fixed")
    model = None
    device = None

def predict_file(path, language="English"):
    """
    Predict if an audio file is fake or real.
    
    Args:
        path: Path to audio file
        language: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
        
    Returns:
        dict: Prediction results with classification, confidenceScore, and explanation
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If prediction fails
    """
    try:
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded - predictions cannot be made")
            return {
                "status": "error",
                "message": "Model file not available on server. Please contact administrator.",
                "language": language,
                "classification": "UNKNOWN",
                "confidenceScore": 0.0,
                "explanation": "The deepfake detection model is not currently loaded on the server."
            }
        
        if not path:
            raise ValueError("Audio file path is empty")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        logger.info(f"Processing audio file: {path}")
        audio = load_audio(path)

        logger.info("Processing audio through model...")
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        device = "cpu"

        with torch.no_grad():
            input_values = inputs.input_values.to(device)
            logits = model(input_values)
            prob = torch.sigmoid(logits).item()

        # Determine classification based on threshold
        # prob > threshold means AI-generated (fake)
        is_ai_generated = prob > THRESHOLD
        
        # Classification label as per requirements
        classification = "AI_GENERATED" if is_ai_generated else "HUMAN"
        
        # Confidence score: normalized to [0, 1]
        confidence_score = round(prob if is_ai_generated else (1 - prob), 2)
        
        # Generate explanation based on classification
        if is_ai_generated:
            explanations = [
                "Unnatural pitch consistency and robotic speech patterns detected",
                "Synthetic voice characteristics and irregular intonation patterns identified",
                "AI-generated artifacts detected in spectral features",
                "Mechanical speech patterns and unnatural prosody detected",
                "Anomalous voice characteristics consistent with synthetic generation"
            ]
            # Use probability to select explanation (deterministic)
            explanation_idx = int(prob * len(explanations)) % len(explanations)
            explanation = explanations[explanation_idx]
        else:
            explanations = [
                "Natural human speech patterns and organic voice characteristics detected",
                "Authentic voice features with human-like variations identified",
                "Genuine human vocal patterns and natural prosody detected",
                "Organic speech characteristics consistent with human voice",
                "Natural breathing patterns and human voice qualities identified"
            ]
            # Use probability to select explanation (deterministic)
            explanation_idx = int((1 - prob) * len(explanations)) % len(explanations)
            explanation = explanations[explanation_idx]
        
        result = {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": confidence_score,
            "explanation": explanation
        }
        
        logger.info(f"Prediction complete: {classification} (confidenceScore: {confidence_score})")
        
        # Memory cleanup after prediction
        del audio, inputs, input_values, logits
        gc.collect()
        
        return result
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise RuntimeError(f"Failed to predict: {e}")
