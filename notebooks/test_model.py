# %% [markdown]
# # Model Evaluation
# 
# This notebook loads a trained model and evaluates it on the test set.

# %%
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from datasets.dataset import get_data_loaders
from models.cnn_model import ImprovedCNN

# %% [markdown]
# ## 1. Load Model and Data

# %%
def load_checkpoint(checkpoint_path):
    """Load a saved model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    # Create model with same configuration
    model = ImprovedCNN(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['config']

# %%
# Set the path to your checkpoint
checkpoint_path = '../checkpoints/best_model_YYYYMMDD_HHMMSS.pth'  # Update this
model, config = load_checkpoint(checkpoint_path)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load test data
_, _, test_loader = get_data_loaders(
    dataset_name=config['dataset'],
    data_dir='../data',
    batch_size=config['batch_size']
)

# %% [markdown]
# ## 2. Evaluate Model

# %%
def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return predictions and true labels."""
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100 * correct / total
    return np.array(all_preds), np.array(all_labels), accuracy

# %%
# Run evaluation
predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
print(f"\nTest Accuracy: {accuracy:.2f}%")

# %% [markdown]
# ## 3. Confusion Matrix

# %%
def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix with nice formatting."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# %%
# Define class names based on dataset
if config['dataset'] == 'cifar10':
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
else:  # cifar100
    # Add CIFAR-100 class names if needed
    class_names = [f"class_{i}" for i in range(100)]

# Plot confusion matrix
plot_confusion_matrix(true_labels, predictions, class_names)

# %% [markdown]
# ## 4. Detailed Classification Report

# %%
# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=class_names)) 