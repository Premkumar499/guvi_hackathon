"""
Download model file from Google Drive if it doesn't exist locally.
"""
import os
import requests
import sys

MODEL_PATH = "models/deepfake_model_v2.pth"
# Extract file ID from your Google Drive link: 1pva0o6QDdFcoq4gC2figHGTIc8ng48Vz
FILE_ID = "1pva0o6QDdFcoq4gC2figHGTIc8ng48Vz"

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive, handling large file confirmation."""
    print(f"📥 Downloading model from Google Drive...")
    print(f"   File ID: {file_id}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Use direct download URL that works for large files
    URL = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    
    print("   Starting download...")
    
    try:
        response = requests.get(URL, stream=True, timeout=300)
        response.raise_for_status()
        
        # Check if we got HTML (error page) instead of file
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print("   ⚠️ Got HTML response, trying alternate method...")
            # Try with session and cookies
            session = requests.Session()
            response = session.get(
                f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t",
                stream=True,
                timeout=300
            )
        
        # Save the file
        print("   Saving file...")
        CHUNK_SIZE = 32768  # 32KB chunks
        downloaded = 0
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress every 50MB
                    if downloaded % (50 * 1024 * 1024) < CHUNK_SIZE:
                        print(f"   Downloaded: {downloaded / (1024*1024):.1f} MB...")
        
        file_size = os.path.getsize(destination)
        
        # Validate file size (should be around 360MB)
        if file_size < 1000000:  # Less than 1MB suggests download failed
            print(f"   ⚠️ Downloaded file too small ({file_size} bytes), likely an error page")
            os.remove(destination)
            raise Exception("Download failed - file too small")
        
        print(f"✅ Model downloaded successfully to {destination}")
        print(f"📊 Final file size: {file_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Download error: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        raise

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
