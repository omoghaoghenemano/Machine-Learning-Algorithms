import torch.nn as nn
import json
import numpy as np
import torch
from torchvision import models
import os
import pickle
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from mlab.neural_networks._mlp import MLPClassifier

def export_architecture(model, path: str):
    """Export MLP architecture to JSON file"""
    arch = []
    # For MLPClassifier only: export Dense layers
    if hasattr(model, 'layers_'):
        for i, layer in enumerate(model.layers_):
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                arch.append({
                    'type': 'Dense',
                    'name': f'layer_{i}',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
    
    # Export classification head
    if hasattr(model, 'out_layer_') and model.out_layer_ is not None:
        arch.append({
            'type': 'Dense',
            'name': 'classifier',
            'in_features': model.out_layer_.in_features,
            'out_features': model.out_layer_.out_features
        })
    
    with open(path, 'w') as f:
        json.dump(arch, f, indent=2)
    print(f"Architecture exported to {path}")

def export_parameters(model, path: str):
    """Export MLP parameters to NPZ file"""
    params = {}
    
    # For MLPClassifier hidden layers
    if hasattr(model, 'layers_'):
        for i, layer in enumerate(model.layers_):
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                params[f'layer{i}.weight'] = layer.weight
                params[f'layer{i}.bias'] = layer.bias
    
    # Export output layer
    if hasattr(model, 'out_layer_') and model.out_layer_ is not None:
        params['classifier.weight'] = model.out_layer_.weight_
        params['classifier.bias'] = model.out_layer_.bias_
    
    np.savez(path, **params)
    print(f"Parameters exported to {path}")
    return params

class AdaptedCNN:
    """
    CNN with frozen ResNet backbone + trainable MLP head
    """
    def __init__(self, mlp_classifier):
        # Frozen ResNet backbone (feature extractor)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.eval()
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Trainable MLP head
        self.mlp_head = mlp_classifier
    
    def predict(self, images):
        """
        End-to-end prediction: Images → Features → Classification
        Args: images (batch, 224, 224, 3)
        Returns: predictions (batch, 45)
        """
        # Extract features using frozen backbone
        features = self._extract_features(images)
        
        # Classify using trained MLP head
        return self.mlp_head.predict(features)
    
    def _extract_features(self, images):
        """Extract features using frozen ResNet backbone"""
        # Convert numpy to torch (N, H, W, C) → (N, C, H, W)
        if isinstance(images, np.ndarray):
            X_torch = torch.tensor(images).permute(0, 3, 1, 2).float()
        else:
            X_torch = images
            
        with torch.no_grad():
            features = self.backbone(X_torch)
            features = features.squeeze()
            if features.ndim == 1:
                features = features.unsqueeze(0)
        
        return features.numpy()

def create_model():
    """
    Loads finetuned_architecture.json and finetuned_parameters.npz, 
    builds an integrated CNN with frozen backbone + trained MLP head.
    Returns a model instance with .predict(batch) method (output shape: (batch, 45)).
    """
    with open('finetuned_architecture.json') as f:
        arch = json.load(f)
    params = np.load('finetuned_parameters.npz')

    # Extract hidden layer sizes from architecture
    hidden_layer_sizes = []
    for spec in arch:
        if spec['type'] == 'Dense' and spec['name'] != 'classifier':
            hidden_layer_sizes.append(spec['out_features'])
    
    # Create MLPClassifier for the head
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation='relu',
        lr=0.003,
        epochs=1,
        random_state=42,
        class_weight='balanced',
    )

    # Extract input size and initialize with dummy data
    input_size = arch[0]['in_features'] if arch else 512
    dummy_X = np.random.randn(2, input_size)
    dummy_y = np.array([0, 1])
    mlp.fit(dummy_X, dummy_y)  # Initialize layers
    
    # Load actual weights
    if hasattr(mlp, 'layers_') and mlp.layers_ is not None:
        for i, layer in enumerate(mlp.layers_):
            w_name = f'layer{i}.weight'
            b_name = f'layer{i}.bias'
            if w_name in params and b_name in params:
                layer.weight = params[w_name]
                layer.bias = params[b_name]
    
    # Load output layer weights
    if hasattr(mlp, 'out_layer_') and mlp.out_layer_ is not None:
        if 'classifier.weight' in params and 'classifier.bias' in params:
            mlp.out_layer_.weight_ = params['classifier.weight']
            mlp.out_layer_.bias_ = params['classifier.bias']

    # Return integrated CNN with frozen backbone + trained head
    return AdaptedCNN(mlp)

# ============ DATA LOADING (ONCE) ============
cache_file = 'resisc45_proper_cached.pkl'

if os.path.exists(cache_file):
    print("Loading cached RESISC45 data...")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    X_train, X_val, X_test, y_train, y_val, y_test = (
        cached_data['X_train'], cached_data['X_val'], cached_data['X_test'],
        cached_data['y_train'], cached_data['y_val'], cached_data['y_test']
    )
else:
    print("Loading and processing RESISC45 dataset...")
    dataset = load_dataset("tanganke/resisc45")
    
    def process_resisc45(dataset_split, max_samples_per_class=100):
        images = []
        labels = []
        class_samples = {}
        for item in dataset_split:
            label = item['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(item)
        
        for label, samples in class_samples.items():
            if max_samples_per_class:
                samples = samples[:max_samples_per_class]
            
            for item in samples:
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((224, 224), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array - mean) / std
                images.append(img_array)
                labels.append(label)
        
        return np.array(images), np.array(labels)
    
    X_all, y_all = process_resisc45(dataset['train'], max_samples_per_class=100)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    cached_data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

# ============ FEATURE EXTRACTION (ONCE) ============
print("Extracting features with ResNet-18...")
resnet18_extractor = models.resnet18(pretrained=True)
resnet18_extractor = torch.nn.Sequential(*list(resnet18_extractor.children())[:-1])
resnet18_extractor.eval()

def extract_features_batch(X_batch, model):
    X_torch = torch.tensor(X_batch).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        features = model(X_torch)
        features = features.squeeze()
        if features.ndim == 1:
            features = features.unsqueeze(0)
    return features.numpy()

def extract_features_in_batches(X, batch_size=32):
    all_features = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        features = extract_features_batch(batch, resnet18_extractor)
        all_features.append(features)
    return np.vstack(all_features)

X_train_features = extract_features_in_batches(X_train)
X_val_features = extract_features_in_batches(X_val)
X_test_features = extract_features_in_batches(X_test)

# ============ TRAIN MLP ============
print("Training MLP classifier...")
mlp = MLPClassifier(
    hidden_layer_sizes=(2048, 1024, 512, 256, 128),
    activation='relu',
    lr=0.003,
    epochs=500,
    random_state=42,
    class_weight='balanced',
)
mlp.fit(X_train_features, y_train)

# ============ EVALUATION ============
y_val_pred = mlp.predict(X_val_features)
val_accuracy = accuracy_score(y_val, y_val_pred)
y_test_pred = mlp.predict(X_test_features)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# ============ EXPORT MODEL ============
export_architecture(mlp, 'finetuned_architecture.json')
export_parameters(mlp, 'finetuned_parameters.npz')

# ============ TEST LOADED MODEL ============
print("\nTesting loaded model...")
model_loaded = create_model()

# Test with original images (not pre-extracted features)
test_batch_images = X_test[:8]  # Use original images, not features
preds = model_loaded.predict(test_batch_images)
print(f"Loaded model prediction shape: {preds.shape}")
print("✅ Model now takes images directly and outputs predictions!")