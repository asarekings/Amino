"""
Utility functions for the Amino crop classification project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import json
import logging
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_submission_file(pixel_ids: List[str], 
                         predictions: np.ndarray,
                         class_names: List[str] = ['cocoa', 'rubber', 'oil'],
                         output_path: str = "submission.csv"):
    """
    Create submission file in required format.
    
    Args:
        pixel_ids: List of unique pixel IDs
        predictions: Prediction probabilities of shape (n_samples, n_classes)
        class_names: List of class names
        output_path: Output file path
    """
    logger.info(f"Creating submission file at {output_path}")
    
    # Create dataframe
    submission_data = {'unique_id': pixel_ids}
    
    for i, class_name in enumerate(class_names):
        submission_data[class_name] = predictions[:, i]
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission file saved with {len(submission_df)} records")
    logger.info(f"Preview:\n{submission_df.head()}")
    
    return submission_df


def evaluate_model_performance(y_true: np.ndarray, 
                             y_pred_proba: np.ndarray,
                             class_names: List[str] = ['cocoa', 'rubber', 'oil']) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics = {
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    # ROC AUC for multiclass
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, 
                                               multi_class='ovr', average='macro')
    except ValueError:
        metrics['roc_auc_macro'] = None
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def plot_training_history(history: Dict[str, List[float]], 
                        save_path: Optional[str] = None):
    """Plot training history for neural network."""
    if 'train_loss' not in history or 'val_loss' not in history:
        logger.warning("No training history to plot")
        return
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if 'train_acc' in history and 'val_acc' in history:
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                         class_names: List[str] = ['cocoa', 'rubber', 'oil'],
                         save_path: Optional[str] = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names: Optional[List[str]] = None,
                          top_n: int = 20, save_path: Optional[str] = None):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} Feature Importances")
    plt.barh(range(len(indices)), importances[indices][::-1])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels: List[str], 
                          class_names: List[str] = ['cocoa', 'rubber', 'oil'],
                          save_path: Optional[str] = None):
    """Plot class distribution."""
    from collections import Counter
    
    counts = Counter(labels)
    
    plt.figure(figsize=(8, 6))
    
    classes = [counts[name] for name in class_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    plt.pie(classes, labels=class_names, autopct='%1.1f%%', colors=colors)
    plt.title('Class Distribution')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_time_series_examples(data: pd.DataFrame, 
                            pixel_ids: List[str],
                            spectral_bands: List[str],
                            save_path: Optional[str] = None):
    """Plot time-series examples for selected pixels."""
    n_pixels = len(pixel_ids)
    n_bands = len(spectral_bands)
    
    fig, axes = plt.subplots(n_pixels, 1, figsize=(12, 3 * n_pixels))
    if n_pixels == 1:
        axes = [axes]
    
    for i, pixel_id in enumerate(pixel_ids):
        pixel_data = data[data['unique_id'] == pixel_id].sort_values('time')
        
        for band in spectral_bands:
            axes[i].plot(pixel_data['time'], pixel_data[band], 
                        label=band, alpha=0.7)
        
        axes[i].set_title(f"Pixel {pixel_id}")
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Reflectance')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time series plot saved to {save_path}")
    
    plt.show()


def save_model_results(results: Dict[str, Any], output_dir: str):
    """Save model results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = os.path.join(output_dir, "metrics.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    with open(metrics_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Model results saved to {output_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")


def get_memory_usage():
    """Get current memory usage."""
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }


def print_system_info():
    """Print system information."""
    import psutil
    import platform
    
    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # GPU information
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.info("GPU: Not available")
    except ImportError:
        logger.info("GPU: PyTorch not available")


class ProgressTracker:
    """Track progress during training and evaluation."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
    def update(self, step: Optional[int] = None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = (self.current_step / self.total_steps) * 100
        logger.info(f"{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps})")
    
    def finish(self):
        """Mark as finished."""
        logger.info(f"{self.description}: Completed!")


def validate_data_format(data: pd.DataFrame, required_columns: List[str]):
    """Validate that data has required format."""
    missing_columns = set(required_columns) - set(data.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("Data format validation passed")


def create_directories(base_path: str):
    """Create necessary directories for outputs."""
    directories = [
        'models',
        'outputs',
        'submissions',
        'logs',
        'plots'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
    
    logger.info(f"Created output directories in {base_path}")