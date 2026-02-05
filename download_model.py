"""
Download model file from Google Drive if it doesn't exist locally.
"""
import os
import requests
import sys
import re

MODEL_PATH = "models/deepfake_model_v2.pth"
# Extract file ID from your Google Drive link: 1pva0o6QDdFcoq4gC2figHGTIc8ng48Vz
FILE_ID = "1pva0o6QDdFcoq4gC2figHGTIc8ng48Vz"

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive, handling large file confirmation."""
    print(f"📥 Downloading model from Google Drive...")
    print(f"   File ID: {file_id}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    
    # First request
    print("   Making initial request...")
    response = session.get(URL, stream=True)
    
    # Check for virus scan warning (large files)
    if 'text/html' in response.headers.get('Content-Type', ''):
        print("   Large file detected, extracting confirmation...")
        
        # Get the HTML content
        html_content = response.text
        
        # Extract confirm and uuid from the HTML form
        confirm_match = re.search(r'name="confirm" value="([^"]+)"', html_content)
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)
        
        if confirm_match:
            confirm_value = confirm_match.group(1)
            print(f"   Found confirmation token: {confirm_value}")
            
            # Build the download URL with confirmation
            params = {
                'id': file_id,
                'export': 'download',
                'confirm': confirm_value
            }
            
            if uuid_match:
                params['uuid'] = uuid_match.group(1)
            
            # Make the actual download request
            print("   Downloading with confirmation...")
            response = session.get(URL, params=params, stream=True)
    
    # Save the file
    print("   Saving file...")
    CHUNK_SIZE = 8192
    downloaded = 0
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Print progress every 10MB
                if downloaded % (10 * 1024 * 1024) < CHUNK_SIZE:
                    print(f"   Downloaded: {downloaded / (1024*1024):.1f} MB...")
    
    file_size = os.path.getsize(destination)
    print(f"✅ Model downloaded successfully to {destination}")
    print(f"📊 Final file size: {file_size / (1024*1024):.2f} MB")

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
