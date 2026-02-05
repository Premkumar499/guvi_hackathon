"""
Download model file from Google Drive if it doesn't exist locally.
"""
import os
import requests
import sys

MODEL_PATH = "models/deepfake_model_v2.pth"
# Extract file ID from your Google Drive link: 1pva0o6QDdFcoq4gC2figHGTIc8ng48Vz
FILE_ID = "1pva0o6QDdFcoq4gC2figHGTIc8ng48Vz"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive."""
    print(f"📥 Downloading model from Google Drive...")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    # Handle the confirmation token for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    # Save the file
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    
    print(f"✅ Model downloaded successfully to {destination}")
    print(f"📊 File size: {os.path.getsize(destination) / (1024*1024):.2f} MB")

def main():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model file already exists at {MODEL_PATH}")
        return
    
    print(f"⚠️ Model file not found at {MODEL_PATH}")
    print(f"📥 Starting download from Google Drive...")
    
    try:
        download_file_from_google_drive(FILE_ID, MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
