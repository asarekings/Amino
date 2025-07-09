"""
Data preprocessing utilities for satellite time-series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteDataProcessor:
    """
    Processes satellite time-series data for crop classification.
    
    Handles:
    - Data loading and parsing
    - Missing data imputation
    - Time-series normalization
    - Train/validation/test splits
    """
    
    def __init__(self, data_path: str):
        """Initialize with path to satellite data CSV."""
        self.data_path = data_path
        self.spectral_bands = [
            'red', 'nir', 'swir16', 'swir22', 'blue', 'green',
            'rededge1', 'rededge2', 'rededge3', 'nir08'
        ]
        self.scaler = StandardScaler()
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load satellite data from CSV file."""
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        # Convert time to datetime
        self.data['time'] = pd.to_datetime(self.data['time'])
        
        logger.info(f"Loaded {len(self.data)} records for {self.data['unique_id'].nunique()} unique pixels")
        return self.data
    
    def handle_missing_data(self, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in spectral bands.
        
        Args:
            method: 'interpolate', 'forward_fill', or 'drop'
        """
        logger.info(f"Handling missing data using method: {method}")
        
        if method == 'interpolate':
            # Group by pixel and interpolate time-series
            for band in self.spectral_bands:
                self.data[band] = self.data.groupby('unique_id')[band].transform(
                    lambda x: x.interpolate(method='linear')
                )
        elif method == 'forward_fill':
            for band in self.spectral_bands:
                self.data[band] = self.data.groupby('unique_id')[band].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        elif method == 'drop':
            self.data = self.data.dropna(subset=self.spectral_bands)
            
        logger.info(f"After handling missing data: {len(self.data)} records")
        return self.data
    
    def normalize_spectral_bands(self, fit_on_train: bool = True) -> pd.DataFrame:
        """
        Normalize spectral band values.
        
        Args:
            fit_on_train: Whether to fit scaler on training data only
        """
        logger.info("Normalizing spectral bands")
        
        if fit_on_train and hasattr(self, 'train_data'):
            # Fit scaler on training data only
            self.scaler.fit(self.train_data[self.spectral_bands])
        else:
            # Fit on all data
            self.scaler.fit(self.data[self.spectral_bands])
        
        # Transform all data
        self.data[self.spectral_bands] = self.scaler.transform(self.data[self.spectral_bands])
        
        return self.data
    
    def ensure_time_series_consistency(self, min_observations: int = 5) -> pd.DataFrame:
        """
        Ensure each pixel has consistent time-series data.
        
        Args:
            min_observations: Minimum number of time observations per pixel
        """
        logger.info(f"Ensuring time-series consistency (min {min_observations} observations)")
        
        # Count observations per pixel
        pixel_counts = self.data.groupby('unique_id').size()
        valid_pixels = pixel_counts[pixel_counts >= min_observations].index
        
        # Filter data to valid pixels only
        self.data = self.data[self.data['unique_id'].isin(valid_pixels)]
        
        # Sort by pixel and time
        self.data = self.data.sort_values(['unique_id', 'time']).reset_index(drop=True)
        
        logger.info(f"After filtering: {len(self.data)} records for {len(valid_pixels)} pixels")
        return self.data
    
    def create_synthetic_labels(self, seed: int = 42) -> pd.DataFrame:
        """
        Create synthetic crop type labels for demonstration.
        In a real scenario, these would come from ground truth data.
        """
        logger.info("Creating synthetic crop type labels")
        
        np.random.seed(seed)
        unique_pixels = self.data['unique_id'].unique()
        
        # Create synthetic labels with some spatial correlation
        labels = np.random.choice(['cocoa', 'rubber', 'oil'], size=len(unique_pixels), 
                                 p=[0.4, 0.35, 0.25])
        
        # Create label mapping
        pixel_labels = pd.DataFrame({
            'unique_id': unique_pixels,
            'crop_type': labels
        })
        
        # Merge with main data
        self.data = self.data.merge(pixel_labels, on='unique_id')
        
        logger.info(f"Label distribution: {self.data.groupby('crop_type')['unique_id'].nunique().to_dict()}")
        return self.data
    
    def create_train_val_test_splits(self, 
                                   train_size: float = 0.7,
                                   val_size: float = 0.15,
                                   test_size: float = 0.15,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets by pixel ID.
        
        Args:
            train_size: Fraction for training
            val_size: Fraction for validation  
            test_size: Fraction for testing
            random_state: Random seed
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
        
        logger.info(f"Creating splits: train={train_size}, val={val_size}, test={test_size}")
        
        # Get unique pixels and their labels
        pixel_data = self.data.groupby('unique_id')['crop_type'].first().reset_index()
        
        # First split: train vs (val + test)
        train_pixels, temp_pixels = train_test_split(
            pixel_data, 
            test_size=(val_size + test_size),
            stratify=pixel_data['crop_type'],
            random_state=random_state
        )
        
        # Second split: val vs test
        val_pixels, test_pixels = train_test_split(
            temp_pixels,
            test_size=test_size/(val_size + test_size),
            stratify=temp_pixels['crop_type'],
            random_state=random_state
        )
        
        # Create data splits
        train_data = self.data[self.data['unique_id'].isin(train_pixels['unique_id'])]
        val_data = self.data[self.data['unique_id'].isin(val_pixels['unique_id'])]
        test_data = self.data[self.data['unique_id'].isin(test_pixels['unique_id'])]
        
        # Store for later use
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        logger.info(f"Split created - Train: {len(train_data)} records ({train_pixels.shape[0]} pixels)")
        logger.info(f"Split created - Val: {len(val_data)} records ({val_pixels.shape[0]} pixels)")
        logger.info(f"Split created - Test: {len(test_data)} records ({test_pixels.shape[0]} pixels)")
        
        return train_data, val_data, test_data
    
    def get_time_series_sequences(self, data: pd.DataFrame, 
                                sequence_length: Optional[int] = None) -> Dict:
        """
        Convert time-series data to sequences for model training.
        
        Args:
            data: DataFrame with time-series data
            sequence_length: Fixed sequence length (None for variable length)
            
        Returns:
            Dictionary with pixel_ids, sequences, and labels
        """
        sequences = []
        pixel_ids = []
        labels = []
        
        # Determine sequence length if not provided
        if sequence_length is None:
            # Use the maximum sequence length in the data
            sequence_lengths = data.groupby('unique_id').size()
            sequence_length = int(sequence_lengths.quantile(0.95))  # Use 95th percentile
            logger.info(f"Using sequence length: {sequence_length}")
        
        for pixel_id in data['unique_id'].unique():
            pixel_data = data[data['unique_id'] == pixel_id].sort_values('time')
            
            # Extract spectral sequence
            sequence = pixel_data[self.spectral_bands].values
            
            if len(sequence) > sequence_length:
                # Truncate to fixed length
                sequence = sequence[:sequence_length]
            elif len(sequence) < sequence_length:
                # Pad with last observation
                last_obs = sequence[-1] if len(sequence) > 0 else np.zeros(len(self.spectral_bands))
                padding = np.tile(last_obs, (sequence_length - len(sequence), 1))
                sequence = np.vstack([sequence, padding])
            
            sequences.append(sequence)
            pixel_ids.append(pixel_id)
            labels.append(pixel_data['crop_type'].iloc[0])
        
        return {
            'pixel_ids': pixel_ids,
            'sequences': np.array(sequences),
            'labels': labels
        }
    
    def process_data(self) -> Tuple[Dict, Dict, Dict]:
        """
        Complete data processing pipeline.
        
        Returns:
            Tuple of (train_dict, val_dict, test_dict) with sequences
        """
        logger.info("Starting complete data processing pipeline")
        
        # Load and preprocess
        self.load_data()
        self.handle_missing_data()
        self.ensure_time_series_consistency()
        self.create_synthetic_labels()
        
        # Create splits
        train_data, val_data, test_data = self.create_train_val_test_splits()
        
        # Normalize (fit on train data)
        self.normalize_spectral_bands(fit_on_train=True)
        
        # Convert to sequences
        train_sequences = self.get_time_series_sequences(self.train_data)
        val_sequences = self.get_time_series_sequences(self.val_data)
        test_sequences = self.get_time_series_sequences(self.test_data)
        
        logger.info("Data processing pipeline completed")
        
        return train_sequences, val_sequences, test_sequences