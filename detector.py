import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import logging
import os

# Disable tokenizers parallelism to save memory
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Detector(nn.Module):
    """
    Deepfake audio detector using Wav2Vec2 encoder.
    Memory-optimized for 512MB environments.
    """
    def __init__(self):
        """
        Initialize the detector model with memory optimizations.
        
        Raises:
            Exception: If model initialization fails
        """
        try:
            super().__init__()
            logger.info("Initializing Wav2Vec2 encoder (memory-optimized)...")
            
            # Load with low_cpu_mem_usage to reduce peak memory
            self.encoder = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            self.classifier = nn.Linear(768, 1)
            
            logger.info("Detector model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Detector: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output logits
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            if x is None:
                raise ValueError("Input tensor is None")
                
            x = self.encoder(x).last_hidden_state
            x = x.mean(dim=1)
            return self.classifier(x)
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise RuntimeError(f"Model forward pass failed: {e}")
