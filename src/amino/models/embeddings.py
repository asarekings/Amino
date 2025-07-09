"""
GeoFM embedding extraction using PRESTO model.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class PrestoEmbeddingExtractor:
    """
    Extract embeddings using PRESTO (Pretrained Remote Sensing Transformer) model.
    
    PRESTO is a foundation model for Earth observation data that can generate
    embeddings from satellite time-series data.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/Presto",
                 device: str = "auto",
                 batch_size: int = 32):
        """
        Initialize PRESTO embedding extractor.
        
        Args:
            model_name: HuggingFace model name for PRESTO
            device: Device to use ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model (fallback to simple encoder if PRESTO not available)
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self):
        """Load PRESTO model or fallback to simple encoder."""
        try:
            # Try to load PRESTO model from HuggingFace
            model = AutoModel.from_pretrained(self.model_name)
            logger.info(f"Loaded PRESTO model: {self.model_name}")
            return model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load PRESTO model: {e}")
            logger.info("Using fallback transformer encoder")
            return self._create_fallback_encoder()
    
    def _create_fallback_encoder(self):
        """Create a simple transformer encoder as fallback."""
        class SimplePrestoEncoder(nn.Module):
            def __init__(self, input_dim=10, hidden_dim=256, num_layers=4, num_heads=8):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(hidden_dim, 256)
                
            def forward(self, x):
                # x shape: (batch_size, sequence_length, num_bands)
                batch_size, seq_len, _ = x.shape
                
                # Project input
                x = self.input_projection(x)
                
                # Add positional encoding
                x = x + self.positional_encoding[:seq_len].unsqueeze(0)
                
                # Apply transformer
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Final projection
                x = self.output_projection(x)
                
                return x
        
        model = SimplePrestoEncoder()
        return model.to(self.device)
    
    def extract_embeddings(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from time-series sequences.
        
        Args:
            sequences: Array of shape (n_samples, sequence_length, n_bands)
            
        Returns:
            embeddings: Array of shape (n_samples, embedding_dim)
        """
        logger.info(f"Extracting embeddings for {len(sequences)} sequences")
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch = sequences[i:i + self.batch_size]
                
                # Convert to tensor
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                # Extract embeddings
                if hasattr(self.model, 'last_hidden_state'):
                    # HuggingFace style model
                    outputs = self.model(batch_tensor)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Custom model
                    batch_embeddings = self.model(batch_tensor)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        
        return embeddings


class ClayEmbeddingExtractor:
    """
    Alternative embedding extractor using Clay model architecture.
    
    Clay is another foundation model for Earth observation data.
    """
    
    def __init__(self, device: str = "auto", batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.batch_size = batch_size
        self.model = self._create_clay_encoder()
        self.model.eval()
        
    def _create_clay_encoder(self):
        """Create Clay-inspired encoder."""
        class ClayEncoder(nn.Module):
            def __init__(self, input_dim=10, hidden_dim=512, num_layers=6):
                super().__init__()
                
                # Convolutional feature extraction
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
                
                # Final layers
                self.layer_norm = nn.LayerNorm(hidden_dim)
                self.output_projection = nn.Linear(hidden_dim, 384)
                
            def forward(self, x):
                # x shape: (batch_size, sequence_length, num_bands)
                batch_size, seq_len, num_bands = x.shape
                
                # Transpose for conv1d: (batch_size, num_bands, sequence_length)
                x = x.transpose(1, 2)
                
                # Apply convolutions
                x = self.conv_layers(x)
                
                # Transpose back: (batch_size, sequence_length, hidden_dim)
                x = x.transpose(1, 2)
                
                # Apply attention
                attn_output, _ = self.attention(x, x, x)
                x = x + attn_output
                x = self.layer_norm(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Final projection
                x = self.output_projection(x)
                
                return x
        
        return ClayEncoder().to(self.device)
    
    def extract_embeddings(self, sequences: np.ndarray) -> np.ndarray:
        """Extract embeddings using Clay model."""
        logger.info(f"Extracting Clay embeddings for {len(sequences)} sequences")
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch = sequences[i:i + self.batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                batch_embeddings = self.model(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        logger.info(f"Extracted Clay embeddings shape: {embeddings.shape}")
        
        return embeddings


class EnsembleEmbeddingExtractor:
    """
    Ensemble of multiple GeoFM models for robust embedding extraction.
    """
    
    def __init__(self, 
                 use_presto: bool = True,
                 use_clay: bool = True,
                 device: str = "auto",
                 batch_size: int = 32):
        """
        Initialize ensemble embedding extractor.
        
        Args:
            use_presto: Whether to use PRESTO model
            use_clay: Whether to use Clay model  
            device: Device to use
            batch_size: Batch size for processing
        """
        self.extractors = []
        
        if use_presto:
            try:
                self.extractors.append(('presto', PrestoEmbeddingExtractor(device=device, batch_size=batch_size)))
                logger.info("Added PRESTO extractor to ensemble")
            except Exception as e:
                logger.warning(f"Could not initialize PRESTO extractor: {e}")
        
        if use_clay:
            try:
                self.extractors.append(('clay', ClayEmbeddingExtractor(device=device, batch_size=batch_size)))
                logger.info("Added Clay extractor to ensemble")
            except Exception as e:
                logger.warning(f"Could not initialize Clay extractor: {e}")
        
        if not self.extractors:
            raise ValueError("No extractors could be initialized")
            
        logger.info(f"Ensemble initialized with {len(self.extractors)} extractors")
    
    def extract_embeddings(self, sequences: np.ndarray, 
                         combination_method: str = "concat") -> np.ndarray:
        """
        Extract embeddings using ensemble of models.
        
        Args:
            sequences: Input sequences
            combination_method: How to combine embeddings ('concat', 'mean', 'stack')
            
        Returns:
            Combined embeddings
        """
        logger.info(f"Extracting ensemble embeddings with {combination_method} combination")
        
        all_embeddings = []
        
        for name, extractor in self.extractors:
            logger.info(f"Extracting embeddings with {name}")
            embeddings = extractor.extract_embeddings(sequences)
            all_embeddings.append(embeddings)
        
        if combination_method == "concat":
            # Concatenate all embeddings
            combined = np.hstack(all_embeddings)
        elif combination_method == "mean":
            # Average all embeddings (assuming same dimension)
            combined = np.mean(all_embeddings, axis=0)
        elif combination_method == "stack":
            # Stack as additional dimension
            combined = np.stack(all_embeddings, axis=-1)
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        
        logger.info(f"Combined embeddings shape: {combined.shape}")
        return combined


def extract_geofm_embeddings(sequences_dict: Dict,
                           extractor_type: str = "ensemble",
                           **kwargs) -> Dict:
    """
    High-level function to extract GeoFM embeddings.
    
    Args:
        sequences_dict: Dictionary with 'sequences' and other data
        extractor_type: Type of extractor ('presto', 'clay', 'ensemble')
        **kwargs: Additional arguments for extractors
        
    Returns:
        Dictionary with embeddings added
    """
    logger.info(f"Extracting GeoFM embeddings using {extractor_type}")
    
    # Filter kwargs for each extractor type
    common_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['device', 'batch_size']}
    
    # Initialize extractor
    if extractor_type == "presto":
        extractor = PrestoEmbeddingExtractor(**common_kwargs)
    elif extractor_type == "clay":
        extractor = ClayEmbeddingExtractor(**common_kwargs)
    elif extractor_type == "ensemble":
        ensemble_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['use_presto', 'use_clay', 'device', 'batch_size']}
        extractor = EnsembleEmbeddingExtractor(**ensemble_kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings(sequences_dict['sequences'])
    
    # Add to dictionary
    result = sequences_dict.copy()
    result['embeddings'] = embeddings
    
    return result