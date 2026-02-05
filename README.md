---
title: Deepfake Voice Detection API
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Deepfake Voice Detection API

Detects AI-generated (deepfake) audio vs human voice.

## API Endpoint

POST `/api/voice-detection`

### Headers
- `Content-Type: application/json`
- `x-api-key: your_api_key`

### Body
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "base64_encoded_audio"
}
```

### Response
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.85,
  "explanation": "Synthetic voice characteristics detected"
}
```
