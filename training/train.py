# training/train.py

import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from datasets.dataset import get_data_loaders
from models.cnn_model import ImprovedCNN

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(config=None):
    if config is None:
        config = {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 0.1,
            'weight_decay': 5e-4,
            'dataset': 'cifar10',
            'model_size': 'medium',
            'data_dir': './data',
            'save_dir': './checkpoints',
            'early_stopping_patience': 10,
            'use_amp': True  # Automatic Mixed Precision
        }
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(save_dir / f'config_{run_id}.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=config['dataset'],
        data_dir=config['data_dir'],
        batch_size=config['batch_size']
    )
    
    # Model, Loss, Optimizer
    model = ImprovedCNN({
        'input_channels': 3,
        'num_classes': 10 if config['dataset'] == 'cifar10' else 100,
        'model_size': config['model_size'],
        'dropout_rate': 0.5  # Adding dropout rate parameter
    }).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    # Automatic Mixed Precision
    scaler = GradScaler('cuda') if config['use_amp'] else None
    
    # Training metrics
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients for every batch
            optimizer.zero_grad()
            
            if config['use_amp']:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(correct / inputs.size(0), inputs.size(0))
            
            # Step the scheduler after optimizer step (for OneCycleLR, we step per batch)
            # Only step after the first optimizer step has been taken
            if epoch > 0 or batch_idx > 0:  # Skip the very first step
                scheduler.step()
            
            pbar.set_postfix({
                'loss': f"{train_loss.avg:.4f}",
                'acc': f"{train_acc.avg*100:.2f}%"
            })
        
        train_losses.append(train_loss.avg)
        
        # Validation phase
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                
                val_loss.update(loss.item(), inputs.size(0))
                val_acc.update(correct / inputs.size(0), inputs.size(0))
        
        val_accuracies.append(val_acc.avg)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg*100:.2f}%")
        print(f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc.avg*100:.2f}%")
        
        # Save best model
        if val_acc.avg > best_val_acc:
            best_val_acc = val_acc.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, save_dir / f'best_model_{run_id}.pth')
            print(f"✅ Saved new best model with validation accuracy: {best_val_acc*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([x * 100 for x in val_accuracies])  # Convert to percentages
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_curves_{run_id}.png')
    
    # Final evaluation on test set
    model.eval()
    test_acc = AverageMeter()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            test_acc.update(correct / inputs.size(0), inputs.size(0))
    
    print(f"\nFinal Test Accuracy: {test_acc.avg*100:.2f}%")
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc.avg,
        'total_epochs': epoch + 1
    }
    
    with open(save_dir / f'results_{run_id}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    train_model()
