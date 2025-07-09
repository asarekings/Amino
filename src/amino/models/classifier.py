"""
Classification models for crop type prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging

logger = logging.getLogger(__name__)


class CropDataset(Dataset):
    """PyTorch Dataset for crop classification."""
    
    def __init__(self, embeddings: np.ndarray, labels: List[str], label_encoder):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor([label_encoder[label] for label in labels])
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class NeuralClassifier(nn.Module):
    """Neural network classifier for crop prediction."""
    
    def __init__(self, input_dim: int, num_classes: int = 3, 
                 hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class CropClassifier:
    """
    Comprehensive crop classification system with multiple model types.
    """
    
    def __init__(self, class_names: List[str] = ['cocoa', 'rubber', 'oil']):
        """
        Initialize crop classifier.
        
        Args:
            class_names: List of crop class names
        """
        self.class_names = class_names
        self.label_encoder = {name: i for i, name in enumerate(class_names)}
        self.label_decoder = {i: name for i, name in enumerate(class_names)}
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')
        
    def _prepare_data(self, data_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare embeddings and labels for training."""
        embeddings = data_dict['embeddings']
        labels = np.array([self.label_encoder[label] for label in data_dict['labels']])
        return embeddings, labels
    
    def train_random_forest(self, train_data: Dict, val_data: Dict, 
                          **kwargs) -> Dict[str, Any]:
        """Train Random Forest classifier."""
        logger.info("Training Random Forest classifier")
        
        X_train, y_train = self._prepare_data(train_data)
        X_val, y_val = self._prepare_data(val_data)
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        # Train model
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        
        # Evaluate
        train_pred_proba = rf.predict_proba(X_train)
        val_pred_proba = rf.predict_proba(X_val)
        
        train_loss = log_loss(y_train, train_pred_proba)
        val_loss = log_loss(y_val, val_pred_proba)
        
        val_pred = rf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.models['random_forest'] = rf
        
        results = {
            'model': rf,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'predictions': val_pred_proba
        }
        
        logger.info(f"Random Forest - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.best_model = ('random_forest', rf)
        
        return results
    
    def train_gradient_boosting(self, train_data: Dict, val_data: Dict,
                              **kwargs) -> Dict[str, Any]:
        """Train Gradient Boosting classifier."""
        logger.info("Training Gradient Boosting classifier")
        
        X_train, y_train = self._prepare_data(train_data)
        X_val, y_val = self._prepare_data(val_data)
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        params.update(kwargs)
        
        # Train model
        gb = GradientBoostingClassifier(**params)
        gb.fit(X_train, y_train)
        
        # Evaluate
        train_pred_proba = gb.predict_proba(X_train)
        val_pred_proba = gb.predict_proba(X_val)
        
        train_loss = log_loss(y_train, train_pred_proba)
        val_loss = log_loss(y_val, val_pred_proba)
        
        val_pred = gb.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.models['gradient_boosting'] = gb
        
        results = {
            'model': gb,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'predictions': val_pred_proba
        }
        
        logger.info(f"Gradient Boosting - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.best_model = ('gradient_boosting', gb)
        
        return results
    
    def train_xgboost(self, train_data: Dict, val_data: Dict,
                     **kwargs) -> Dict[str, Any]:
        """Train XGBoost classifier."""
        logger.info("Training XGBoost classifier")
        
        X_train, y_train = self._prepare_data(train_data)
        X_val, y_val = self._prepare_data(val_data)
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        params.update(kwargs)
        
        # Train model
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     verbose=False)
        
        # Evaluate
        train_pred_proba = xgb_model.predict_proba(X_train)
        val_pred_proba = xgb_model.predict_proba(X_val)
        
        train_loss = log_loss(y_train, train_pred_proba)
        val_loss = log_loss(y_val, val_pred_proba)
        
        val_pred = xgb_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.models['xgboost'] = xgb_model
        
        results = {
            'model': xgb_model,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'predictions': val_pred_proba
        }
        
        logger.info(f"XGBoost - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.best_model = ('xgboost', xgb_model)
        
        return results
    
    def train_lightgbm(self, train_data: Dict, val_data: Dict,
                      **kwargs) -> Dict[str, Any]:
        """Train LightGBM classifier."""
        logger.info("Training LightGBM classifier")
        
        X_train, y_train = self._prepare_data(train_data)
        X_val, y_val = self._prepare_data(val_data)
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        params.update(kwargs)
        
        # Train model
        lgb_model = lgb.LGBMClassifier(**params)
        lgb_model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        # Evaluate
        train_pred_proba = lgb_model.predict_proba(X_train)
        val_pred_proba = lgb_model.predict_proba(X_val)
        
        train_loss = log_loss(y_train, train_pred_proba)
        val_loss = log_loss(y_val, val_pred_proba)
        
        val_pred = lgb_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.models['lightgbm'] = lgb_model
        
        results = {
            'model': lgb_model,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'predictions': val_pred_proba
        }
        
        logger.info(f"LightGBM - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.best_model = ('lightgbm', lgb_model)
        
        return results
    
    def train_neural_network(self, train_data: Dict, val_data: Dict,
                           epochs: int = 100, batch_size: int = 64,
                           learning_rate: float = 0.001,
                           device: str = "auto", **kwargs) -> Dict[str, Any]:
        """Train Neural Network classifier."""
        logger.info("Training Neural Network classifier")
        
        # Set device
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        X_train, y_train = self._prepare_data(train_data)
        X_val, y_val = self._prepare_data(val_data)
        
        # Create datasets and loaders
        train_dataset = CropDataset(X_train, train_data['labels'], self.label_encoder)
        val_dataset = CropDataset(X_val, val_data['labels'], self.label_encoder)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = NeuralClassifier(
            input_dim=X_train.shape[1],
            num_classes=len(self.class_names),
            **kwargs
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            all_pred_proba = []
            all_labels = []
            
            with torch.no_grad():
                for batch_embeddings, batch_labels in val_loader:
                    batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
                    
                    outputs = model(batch_embeddings)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    pred_proba = torch.softmax(outputs, dim=1)
                    all_pred_proba.append(pred_proba.cpu().numpy())
                    all_labels.append(batch_labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model state
                best_model_state = model.state_dict().copy()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_pred_proba = []
            for batch_embeddings, _ in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                outputs = model(batch_embeddings)
                pred_proba = torch.softmax(outputs, dim=1)
                val_pred_proba.append(pred_proba.cpu().numpy())
        
        val_pred_proba = np.vstack(val_pred_proba)
        val_loss_final = log_loss(y_val, val_pred_proba)
        val_pred = np.argmax(val_pred_proba, axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.models['neural_network'] = model
        
        results = {
            'model': model,
            'train_loss': train_losses[-1],
            'val_loss': val_loss_final,
            'val_accuracy': val_acc,
            'predictions': val_pred_proba,
            'train_history': {'train_loss': train_losses, 'val_loss': val_losses}
        }
        
        logger.info(f"Neural Network - Final Val Loss: {val_loss_final:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss_final < self.best_score:
            self.best_score = val_loss_final
            self.best_model = ('neural_network', model)
        
        return results
    
    def train_ensemble(self, train_data: Dict, val_data: Dict) -> Dict[str, Any]:
        """Train ensemble of all models."""
        logger.info("Training ensemble of models")
        
        # Train individual models
        models_results = {}
        
        models_results['rf'] = self.train_random_forest(train_data, val_data)
        models_results['gb'] = self.train_gradient_boosting(train_data, val_data)
        models_results['xgb'] = self.train_xgboost(train_data, val_data)
        models_results['lgb'] = self.train_lightgbm(train_data, val_data)
        models_results['nn'] = self.train_neural_network(train_data, val_data)
        
        # Ensemble predictions (simple averaging)
        ensemble_pred_proba = np.mean([
            results['predictions'] for results in models_results.values()
        ], axis=0)
        
        X_val, y_val = self._prepare_data(val_data)
        ensemble_loss = log_loss(y_val, ensemble_pred_proba)
        ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        
        logger.info(f"Ensemble - Val Loss: {ensemble_loss:.4f}, Val Acc: {ensemble_acc:.4f}")
        
        if ensemble_loss < self.best_score:
            self.best_score = ensemble_loss
            self.best_model = ('ensemble', models_results)
        
        return {
            'ensemble_predictions': ensemble_pred_proba,
            'ensemble_loss': ensemble_loss,
            'ensemble_accuracy': ensemble_acc,
            'individual_results': models_results
        }
    
    def predict(self, test_data: Dict, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            test_data: Test data dictionary
            model_name: Specific model name to use (None for best model)
            
        Returns:
            Prediction probabilities
        """
        X_test, _ = self._prepare_data(test_data)
        
        if model_name is None:
            # Use best model
            model_name, model = self.best_model
        else:
            model = self.models[model_name]
        
        if model_name == 'ensemble':
            # Ensemble prediction
            predictions = []
            for name, results in model['individual_results'].items():
                if name == 'nn':
                    # Neural network prediction
                    nn_model = results['model']
                    nn_model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.FloatTensor(X_test)
                        if next(nn_model.parameters()).is_cuda:
                            X_test_tensor = X_test_tensor.cuda()
                        outputs = nn_model(X_test_tensor)
                        pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                else:
                    pred_proba = results['model'].predict_proba(X_test)
                predictions.append(pred_proba)
            
            return np.mean(predictions, axis=0)
        
        elif model_name == 'neural_network':
            # Neural network prediction
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                if next(model.parameters()).is_cuda:
                    X_test_tensor = X_test_tensor.cuda()
                outputs = model(X_test_tensor)
                return torch.softmax(outputs, dim=1).cpu().numpy()
        
        else:
            # Scikit-learn style model
            return model.predict_proba(X_test)
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models."""
        summary_data = []
        
        for name, model in self.models.items():
            if hasattr(model, 'score'):
                summary_data.append({
                    'Model': name,
                    'Best': name == self.best_model[0] if self.best_model else False
                })
        
        return pd.DataFrame(summary_data)