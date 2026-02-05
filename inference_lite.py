"""
Lightweight inference module for deepfake detection.
Uses simple audio features instead of heavy ML models to fit in 512MB RAM.
"""
import os
import logging
from simple_detector import analyze_audio_simple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("✅ Using lightweight detector (memory optimized for free hosting)")

def predict_file(path: str, language: str = "English") -> dict:
    """
    Predict if an audio file is AI-generated or human.
    
    Args:
        path: Path to audio file
        language: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
        
    Returns:
        dict: Prediction results with classification, confidenceScore, and explanation
    """
    try:
        if not path:
            raise ValueError("Audio file path is empty")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        logger.info(f"Processing {language} audio file: {path}")
        
        # Analyze using lightweight method
        result = analyze_audio_simple(path)
        
        is_ai = result["is_ai_generated"]
        confidence = result["confidence"]
        features = result["features"]
        
        # Classification
        classification = "AI_GENERATED" if is_ai else "HUMAN"
        
        # Generate explanation
        if is_ai:
            explanations = [
                f"Detected {features['pitch_consistency'].lower()} pitch consistency and {features['energy_uniformity'].lower()} energy uniformity typical of synthetic voice",
                f"Audio analysis shows {features['zero_crossing_pattern'].lower()} zero-crossing patterns consistent with AI-generated speech",
                f"Spectral features and {features['pitch_consistency'].lower()} pitch variation suggest synthetic voice generation",
                "Voice characteristics indicate artificial speech synthesis with mechanical patterns",
                "Analysis reveals audio properties commonly associated with AI voice generation"
            ]
            explanation_idx = int(confidence * len(explanations)) % len(explanations)
            explanation = explanations[explanation_idx]
        else:
            explanations = [
                f"Natural voice patterns detected with {features['pitch_consistency'].lower()} pitch consistency typical of human speech",
                f"Audio shows {features['zero_crossing_pattern'].lower()} zero-crossing patterns consistent with organic human voice",
                f"Spectral analysis and {features['energy_uniformity'].lower()} energy variation indicate authentic human speech",
                "Voice characteristics match natural human speech patterns with organic variations",
                "Analysis confirms audio properties associated with genuine human voice"
            ]
            explanation_idx = int((1 - confidence) * len(explanations)) % len(explanations)
            explanation = explanations[explanation_idx]
        
        response = {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }
        
        logger.info(f"Prediction: {classification} (confidence: {confidence})")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "language": language,
            "classification": "UNKNOWN",
            "confidenceScore": 0.0,
            "explanation": "Audio file could not be processed"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "language": language,
            "classification": "UNKNOWN",
            "confidenceScore": 0.0,
            "explanation": "An error occurred during audio analysis"
        }
