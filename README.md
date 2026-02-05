# Deepfake Voice Detection API

**Lightweight Version - Optimized for Free Hosting (<512MB RAM)**

## API Endpoint
```
POST https://guvi-hackathon-no3n.onrender.com/api/voice-detection
```

## Features
✅ Supports 5 languages: Tamil, English, Hindi, Malayalam, Telugu
✅ Base64 MP3 audio input
✅ API key authentication
✅ Returns AI_GENERATED or HUMAN classification
✅ Memory optimized using audio feature analysis

## Note
This is a lightweight version using audio feature analysis (pitch, spectral, energy patterns) instead of heavy ML models to fit within free hosting constraints (512MB RAM). The API structure fully complies with hackathon requirements.

## Request Format
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_AUDIO_HERE"
}
```

## Response Format
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.85,
  "explanation": "Audio analysis shows uniform zero-crossing patterns consistent with AI-generated speech"
}
```

## API Key
Header: `x-api-key: deepfake_detection_api_key_2026`
