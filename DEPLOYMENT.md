# Deepfake Voice Detection API - Render Deployment Guide

## 🚀 Quick Deploy to Render

### Prerequisites
- GitHub account
- Render account (free): https://render.com

### Step 1: Push to GitHub

```bash
cd "/home/premkumar/Downloads/final project/deepfake_api"

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Ready for Render deployment"

# Push to GitHub
git remote add origin https://github.com/Premkumar499/guvi_hackathon.git
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to https://render.com and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `Premkumar499/guvi_hackathon`
4. Configure:
   - **Name**: `deepfake-voice-api`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Instance Type**: `Free`

5. Add Environment Variable:
   - Key: `API_KEY`
   - Value: `deepfake_detection_api_key_2026`

6. Click **"Create Web Service"**

### Step 3: Wait for Deployment
- Initial deployment takes 5-10 minutes
- You'll get a URL like: `https://deepfake-voice-api.onrender.com`

---

## 🧪 Test Your Deployed API

### Using cURL:
```bash
curl -X POST https://your-app.onrender.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: deepfake_detection_api_key_2026" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_AUDIO_HERE"
  }'
```

### Expected Response:
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

---

## 📋 API Endpoints

### 1. Health Check
```
GET https://your-app.onrender.com/
```

### 2. Voice Detection
```
POST https://your-app.onrender.com/api/voice-detection
Headers:
  - Content-Type: application/json
  - x-api-key: deepfake_detection_api_key_2026

Body:
{
  "language": "Tamil|English|Hindi|Malayalam|Telugu",
  "audioFormat": "mp3",
  "audioBase64": "base64_encoded_mp3_audio"
}
```

---

## ⚠️ Important Notes

1. **Free Tier Limitations**:
   - Service spins down after 15 minutes of inactivity
   - First request after idle takes 30-60 seconds to wake up
   - 750 hours/month free

2. **Model Loading**:
   - First request after deployment takes ~30 seconds (model loading)
   - Subsequent requests are fast

3. **Audio Size Limit**:
   - Maximum: 10MB base64 encoded audio

4. **API Key**:
   - Change the default API key for production use
   - Update in Render dashboard: Settings → Environment Variables

---

## 🔧 Troubleshooting

### If deployment fails:
1. Check build logs in Render dashboard
2. Ensure all files are committed to GitHub
3. Verify requirements.txt has all dependencies

### If API returns errors:
1. Check Render logs: Dashboard → Logs
2. Verify model file is included in repository
3. Test API key authentication

---

## 📝 Files Structure

```
deepfake_api/
├── app.py              # Flask API server
├── audio_utils.py      # Audio processing
├── config.py           # Configuration
├── detector.py         # Model architecture
├── inference.py        # Prediction logic
├── requirements.txt    # Python dependencies
├── Procfile           # Render start command
├── render.yaml        # Render configuration
├── runtime.txt        # Python version
├── models/            # Trained model files
└── temp/              # Temporary audio files
```

---

## 🎯 Competition Submission

**Your API URL**: `https://your-app.onrender.com/api/voice-detection`

**Authentication**: `x-api-key: deepfake_detection_api_key_2026`

**Supported Languages**: Tamil, English, Hindi, Malayalam, Telugu

**Response Time**: ~2-5 seconds per request

---

## 📞 Support

For issues, check:
- Render logs
- GitHub repository
- Model files integrity

Good luck with the hackathon! 🚀
