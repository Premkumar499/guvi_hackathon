"""
Lightweight deepfake detector using simple audio features.
Memory efficient - no heavy ML models required.
"""
import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_audio_simple(audio_path: str) -> dict:
    """
    Analyze audio using lightweight features.
    Returns prediction based on simple heuristics.
    
    Note: This is a lightweight version for demo purposes.
    Full ML model requires more memory than free hosting allows.
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30)
        
        # Extract simple features
        # 1. Zero crossing rate (voice naturalness indicator)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # 3. Energy and rhythm
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # 4. Pitch consistency
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # Simple scoring algorithm
        # AI voices tend to have:
        # - More consistent pitch (lower std)
        # - More uniform energy (lower rms_std)
        # - Less natural zero-crossing patterns
        
        score = 0
        
        # Pitch consistency check (AI = more consistent)
        if pitch_std < 20:  # Very consistent pitch
            score += 0.3
        
        # Energy uniformity (AI = more uniform)
        if rms_std < 0.05:
            score += 0.25
        
        # Zero crossing rate (AI = less natural)
        if zcr_std < 0.05:
            score += 0.25
        
        # Spectral characteristics
        spectral_mean = np.mean(spectral_centroids)
        if 1000 < spectral_mean < 3000:  # AI often in this range
            score += 0.2
        
        # Normalize score to 0-1
        confidence = min(max(score, 0.0), 1.0)
        
        # Classification
        threshold = 0.5
        is_ai = confidence >= threshold
        
        result = {
            "is_ai_generated": is_ai,
            "confidence": round(confidence, 2),
            "features": {
                "pitch_consistency": f"{'High' if pitch_std < 20 else 'Normal'}",
                "energy_uniformity": f"{'High' if rms_std < 0.05 else 'Normal'}",
                "zero_crossing_pattern": f"{'Uniform' if zcr_std < 0.05 else 'Natural'}"
            }
        }
        
        logger.info(f"Analysis complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
