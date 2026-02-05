# Alternative Deployment Options

## ⚠️ Model File Issue
Your model file (361MB) is too large for direct GitHub push.

## 🔧 Solutions:

### Option 1: Use Google Drive + Render (Recommended)

1. **Upload model to Google Drive**:
   - Upload `models/deepfake_model_v2.pth` to Google Drive
   - Make it publicly accessible
   - Get shareable link

2. **Download model on Render startup**:
   
Update `render.yaml`:
```yaml
services:
  - type: web
    name: deepfake-voice-detection-api
    env: python
    buildCommand: |
      pip install -r requirements.txt
      pip install gdown
      mkdir -p models
      gdown --id YOUR_GOOGLE_DRIVE_FILE_ID -O models/deepfake_model_v2.pth
    startCommand: gunicorn app:app
    plan: free
```

### Option 2: Hugging Face Hub

1. Upload model to Hugging Face
2. Download in code during initialization

### Option 3: Smaller Repository Approach

1. Remove model from git:
```bash
git rm --cached models/deepfake_model_v2.pth
git commit -m "Remove large model file"
```

2. Push without model:
```bash
git push origin main
```

3. Upload model separately via Render dashboard or download script

### Option 4: Railway (500MB limit)

Use Railway instead of Render - they support larger files:
- https://railway.app
- Supports files up to 500MB
- Similar to Render deployment

---

## 📦 Quick Fix for GitHub

```bash
cd "/home/premkumar/Downloads/final project/deepfake_api"

# Add .gitignore
git add .gitignore

# Remove model from staging
git rm --cached -r models/

# Commit
git commit -m "Ignore large model files"

# Push code only
git push origin main
```

Then deploy model separately using Option 1 above.
