"""
Lightweight Deepfake Detector using MFCC features.
Memory footprint: ~50MB (fits in 512MB free tier)
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature extraction parameters
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 16000
MAX_LEN = SAMPLE_RATE * 2  # 2 seconds


class LightweightDetector(nn.Module):
    """
    Lightweight deepfake detector using MFCC features.
    Much smaller than Wav2Vec2 (~1MB vs 360MB).
    """
    def __init__(self):
        super().__init__()
        
        # CNN for MFCC features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: MFCC features [batch, 1, n_mfcc, time]
        Returns:
            logits [batch, 1]
        """
        x = self.conv_layers(x)
        return self.classifier(x)


def extract_mfcc(audio, sr=SAMPLE_RATE):
    """Extract MFCC features from audio."""
    # Ensure correct length
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # Add delta features
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack features
    features = np.stack([mfcc, delta, delta2], axis=0)
    
    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


def create_lightweight_model():
    """Create and return the lightweight model."""
    model = LightweightDetector()
    model.eval()
    return model
