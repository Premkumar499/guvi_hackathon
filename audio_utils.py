import base64
import librosa
import numpy as np
import os
import uuid
import logging
from config import SAMPLE_RATE, MAX_LEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def decode_base64_audio(b64_string):
    """
    Decode base64 audio string and save to temporary file.
    Accepts both raw base64 and data URI format.
    Supports MP3, WAV, and other audio formats.
    
    Args:
        b64_string: Base64 encoded MP3/audio data (with or without data URI prefix)
        
    Returns:
        str: Path to temporary audio file
        
    Raises:
        ValueError: If base64 string is invalid
        IOError: If file cannot be written
    """
    try:
        if not b64_string:
            raise ValueError("Empty base64 audio string provided")
        
        # Remove data URI prefix if present (e.g., "data:audio/mp3;base64," or "data:audio/wav;base64,")
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]  # Only split on first comma

        # Clean up the base64 string - remove ALL whitespace characters
        import re
        b64_string = re.sub(r'\s+', '', b64_string)
        
        # Convert URL-safe base64 to standard base64
        b64_string = b64_string.replace('-', '+').replace('_', '/')
        
        # Remove any existing padding first, then add correct padding
        b64_string = b64_string.rstrip('=')
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

        # Validate base64 characters before decoding
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        invalid_chars = set(b64_string) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid characters in base64 string: {invalid_chars}")

        audio_bytes = base64.b64decode(b64_string, validate=True)
        
        if len(audio_bytes) == 0:
            raise ValueError("Decoded audio data is empty")
        
        # Generate unique filename - using .mp3 extension to preserve format
        # librosa can handle MP3, WAV, and other formats automatically
        filename = f"{uuid.uuid4().hex}.mp3"
        path = os.path.join(TEMP_DIR, filename)
        
        with open(path, "wb") as f:
            f.write(audio_bytes)
            
        logger.info(f"Successfully decoded and saved audio to {path}")
        return path
        
    except base64.binascii.Error as e:
        logger.error(f"Invalid base64 string: {e}")
        raise ValueError(f"Invalid base64 audio data: {e}")
    except IOError as e:
        logger.error(f"Failed to write audio file: {e}")
        raise IOError(f"Cannot write temporary audio file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in decode_base64_audio: {e}")
        raise

def load_audio(path):
    """
    Load and preprocess audio file.
    
    Args:
        path: Path to audio file
        
    Returns:
        np.ndarray: Preprocessed audio array
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio cannot be loaded or processed
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
            
        if os.path.getsize(path) == 0:
            raise ValueError(f"Audio file is empty: {path}")
            
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        
        if len(audio) == 0:
            raise ValueError("Loaded audio has no samples")

        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
            logger.info(f"Audio trimmed to {MAX_LEN} samples")
        else:
            audio = np.pad(audio, (0, MAX_LEN-len(audio)))
            logger.info(f"Audio padded to {MAX_LEN} samples")

        return audio.astype(np.float32)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error in load_audio: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading audio file {path}: {e}")
        raise ValueError(f"Failed to load audio: {e}")
