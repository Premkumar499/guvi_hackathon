from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_lite import predict_file
from audio_utils import decode_base64_audio
import os
import logging
import traceback
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MAX_BASE64_SIZE = 15_000_000   # ~10MB in base64

# API Key for authentication
API_KEY = os.environ.get("API_KEY", "deepfake_detection_api_key_2026")

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "API key is required. Provide it via x-api-key header."
            }), 401
        
        if api_key != API_KEY:
            return jsonify({
                "status": "error",
                "message": "Invalid API key"
            }), 403
        
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def home():
    """Health check endpoint."""
    return {
        "status": "Deepfake Voice Detection API Running",
        "version": "1.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "endpoints": {
            "/": "Health check",
            "/api/voice-detection": "POST - Audio deepfake detection (requires API key)"
        },
        "authentication": "API key required via x-api-key header"
    }

@app.route("/api/voice-detection", methods=["POST"])
@require_api_key
def voice_detection():
    """
    Predict if uploaded audio is AI-generated or human.
    
    Expected JSON payload:
        {
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": "base64_encoded_mp3_audio"
        }
        
    Returns:
        JSON response with classification (AI_GENERATED/HUMAN) and confidenceScore (0.0-1.0)
    """
    temp_path = None
    
    try:
        # Validate request
        if not request.json:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        data = request.json
        
        # Validate required fields
        if "language" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'language' field in request"
            }), 400
        
        if "audioFormat" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'audioFormat' field in request"
            }), 400
            
        if "audioBase64" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'audioBase64' field in request"
            }), 400
        
        language = data["language"]
        audio_format = data["audioFormat"]
        b64_audio = data["audioBase64"]
        
        # Validate language
        supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        if language not in supported_languages:
            return jsonify({
                "status": "error",
                "message": f"Unsupported language. Must be one of: {', '.join(supported_languages)}"
            }), 400
        
        # Validate audio format
        if audio_format.lower() != "mp3":
            return jsonify({
                "status": "error",
                "message": "Only mp3 format is supported"
            }), 400
        
        if not b64_audio:
            return jsonify({
                "status": "error",
                "message": "Empty audio data provided"
            }), 400
        
        # Check size limit
        if len(b64_audio) > MAX_BASE64_SIZE:
            return jsonify({
                "status": "error",
                "message": "Audio data too large (max 10MB)"
            }), 413
        
        logger.info(f"Received prediction request for {language} audio")
        
        # Decode audio
        temp_path = decode_base64_audio(b64_audio)
        
        # Predict
        result = predict_file(temp_path, language)
        
        # Clean up
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info("Temporary file cleaned up")

        return jsonify(result), 200

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            "status": "error",
            "message": f"Invalid input: {str(e)}"
        }), 400
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            "status": "error",
            "message": f"File error: {str(e)}"
        }), 404
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
