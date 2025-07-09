"""
Main pipeline for Amino crop classification using GeoFMs.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from amino.data import SatelliteDataProcessor
from amino.models import extract_geofm_embeddings, CropClassifier
from amino.utils import (
    setup_logging, create_submission_file, evaluate_model_performance,
    plot_confusion_matrix, plot_class_distribution, save_model_results,
    create_directories, print_system_info
)

import logging
logger = logging.getLogger(__name__)


def main(config_path: str = None):
    """Main pipeline execution."""
    
    # Default configuration
    config = {
        "data": {
            "data_path": "test.csv",
            "min_observations": 5,
            "normalization_method": "interpolate",
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_state": 42
        },
        "embeddings": {
            "extractor_type": "ensemble",
            "use_presto": True,
            "use_clay": True,
            "batch_size": 32,
            "device": "auto"
        },
        "models": {
            "train_models": ["random_forest", "xgboost", "lightgbm", "neural_network"],
            "ensemble": True,
            "hyperparameter_tuning": False
        },
        "output": {
            "output_dir": "outputs",
            "submission_file": "submission.csv",
            "save_models": True,
            "create_plots": True
        },
        "logging": {
            "level": "INFO",
            "log_file": "logs/amino.log"
        }
    }
    
    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
        
        # Update config with custom values
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        config = update_dict(config, custom_config)
    
    # Setup logging and directories
    create_directories(".")
    setup_logging(
        log_level=config["logging"]["level"], 
        log_file=config["logging"]["log_file"]
    )
    
    logger.info("="*60)
    logger.info("AMINO CROP CLASSIFICATION PIPELINE")
    logger.info("="*60)
    
    # Print system information
    print_system_info()
    
    try:
        # Step 1: Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        logger.info("-" * 30)
        
        processor = SatelliteDataProcessor(config["data"]["data_path"])
        
        # Process data
        train_data, val_data, test_data = processor.process_data()
        
        logger.info(f"Training samples: {len(train_data['sequences'])}")
        logger.info(f"Validation samples: {len(val_data['sequences'])}")
        logger.info(f"Test samples: {len(test_data['sequences'])}")
        
        # Create class distribution plot
        if config["output"]["create_plots"]:
            plot_class_distribution(
                train_data['labels'], 
                save_path=os.path.join(config["output"]["output_dir"], "plots", "class_distribution.png")
            )
        
        # Step 2: Embedding Extraction
        logger.info("\nStep 2: Embedding Extraction using GeoFMs")
        logger.info("-" * 45)
        
        # Extract embeddings for all datasets
        train_embeddings = extract_geofm_embeddings(
            train_data, 
            **config["embeddings"]
        )
        
        val_embeddings = extract_geofm_embeddings(
            val_data, 
            **config["embeddings"]
        )
        
        test_embeddings = extract_geofm_embeddings(
            test_data, 
            **config["embeddings"]
        )
        
        logger.info(f"Training embeddings shape: {train_embeddings['embeddings'].shape}")
        logger.info(f"Validation embeddings shape: {val_embeddings['embeddings'].shape}")
        logger.info(f"Test embeddings shape: {test_embeddings['embeddings'].shape}")
        
        # Step 3: Model Training
        logger.info("\nStep 3: Model Training")
        logger.info("-" * 25)
        
        classifier = CropClassifier()
        
        model_results = {}
        
        # Train individual models
        for model_name in config["models"]["train_models"]:
            logger.info(f"Training {model_name}")
            
            if model_name == "random_forest":
                results = classifier.train_random_forest(train_embeddings, val_embeddings)
            elif model_name == "gradient_boosting":
                results = classifier.train_gradient_boosting(train_embeddings, val_embeddings)
            elif model_name == "xgboost":
                results = classifier.train_xgboost(train_embeddings, val_embeddings)
            elif model_name == "lightgbm":
                results = classifier.train_lightgbm(train_embeddings, val_embeddings)
            elif model_name == "neural_network":
                results = classifier.train_neural_network(train_embeddings, val_embeddings)
            
            model_results[model_name] = results
        
        # Train ensemble if requested
        if config["models"]["ensemble"]:
            logger.info("Training ensemble model")
            ensemble_results = classifier.train_ensemble(train_embeddings, val_embeddings)
            model_results["ensemble"] = ensemble_results
        
        # Step 4: Model Evaluation
        logger.info("\nStep 4: Model Evaluation")
        logger.info("-" * 25)
        
        best_model_name = classifier.best_model[0] if classifier.best_model else "ensemble"
        logger.info(f"Best model: {best_model_name} (Val Loss: {classifier.best_score:.4f})")
        
        # Evaluate best model on validation set
        val_predictions = classifier.predict(val_embeddings)
        val_metrics = evaluate_model_performance(
            np.array([classifier.label_encoder[label] for label in val_embeddings['labels']]),
            val_predictions
        )
        
        logger.info(f"Validation Metrics:")
        logger.info(f"  Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {val_metrics['f1_macro']:.4f}")
        
        # Create confusion matrix plot
        if config["output"]["create_plots"]:
            plot_confusion_matrix(
                val_metrics['confusion_matrix'],
                save_path=os.path.join(config["output"]["output_dir"], "plots", "confusion_matrix.png")
            )
        
        # Step 5: Generate Predictions and Submission
        logger.info("\nStep 5: Generate Test Predictions")
        logger.info("-" * 35)
        
        test_predictions = classifier.predict(test_embeddings)
        
        # Create submission file
        submission_path = os.path.join(config["output"]["output_dir"], config["output"]["submission_file"])
        submission_df = create_submission_file(
            test_embeddings['pixel_ids'],
            test_predictions,
            output_path=submission_path
        )
        
        # Step 6: Save Results
        logger.info("\nStep 6: Save Results")
        logger.info("-" * 20)
        
        # Prepare results summary
        results_summary = {
            "data_info": {
                "train_samples": len(train_embeddings['sequences']),
                "val_samples": len(val_embeddings['sequences']),
                "test_samples": len(test_embeddings['sequences']),
                "embedding_dim": train_embeddings['embeddings'].shape[1]
            },
            "best_model": {
                "name": best_model_name,
                "val_loss": classifier.best_score
            },
            "validation_metrics": val_metrics,
            "config": config
        }
        
        # Save results
        results_path = os.path.join(config["output"]["output_dir"], "results_summary.json")
        save_model_results(results_summary, config["output"]["output_dir"])
        
        logger.info("\nPipeline completed successfully!")
        logger.info(f"Results saved to: {config['output']['output_dir']}")
        logger.info(f"Submission file: {submission_path}")
        
        return {
            "classifier": classifier,
            "results": model_results,
            "metrics": val_metrics,
            "submission": submission_df
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amino Crop Classification Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    main(args.config)