"""
Download model file from Hugging Face Hub if it doesn't exist locally.
"""
import os
import sys

MODEL_PATH = "models/deepfake_model_v2.pth"
HF_REPO_ID = "prem678/deepfake-detection-model"
HF_FILENAME = "deepfake_model_v2.pth"

def download_from_huggingface(repo_id, filename, destination):
    """Download a file from Hugging Face Hub."""
    print(f"📥 Downloading model from Hugging Face Hub...")
    print(f"   Repo: {repo_id}")
    print(f"   File: {filename}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        
        print("   Starting download...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        
        # Move to correct location if needed
        if downloaded_path != destination and os.path.exists(downloaded_path):
            import shutil
            shutil.move(downloaded_path, destination)
        
        file_size = os.path.getsize(destination)
        print(f"✅ Model downloaded successfully to {destination}")
        print(f"📊 Final file size: {file_size / (1024*1024):.2f} MB")
        
    except ImportError:
        # Fallback: direct HTTP download without huggingface_hub
        print("   huggingface_hub not installed, using direct download...")
        import requests
        
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        print(f"   URL: {url}")
        
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        CHUNK_SIZE = 32768
        downloaded = 0
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (50 * 1024 * 1024) < CHUNK_SIZE:
                        print(f"   Downloaded: {downloaded / (1024*1024):.1f} MB...")
        
        file_size = os.path.getsize(destination)
        print(f"✅ Model downloaded successfully to {destination}")
        print(f"📊 Final file size: {file_size / (1024*1024):.2f} MB")

def main():
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 100_000_000:  # >100MB means valid model
            print(f"✅ Model file already exists at {MODEL_PATH} ({file_size/(1024*1024):.1f} MB)")
            return
        else:
            print(f"⚠️ Model file exists but is too small ({file_size} bytes), re-downloading...")
            os.remove(MODEL_PATH)
    
    print(f"⚠️ Model file not found at {MODEL_PATH}")
    print(f"📥 Starting download from Hugging Face Hub...")
    
    try:
        download_from_huggingface(HF_REPO_ID, HF_FILENAME, MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
