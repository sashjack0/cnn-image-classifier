# models/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A convolutional block with optional residual connection"""
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)
        
        # Residual connection if input and output channels differ
        if use_residual and in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        
        if self.use_residual:
            out += identity
        
        out = F.relu(out)
        return F.max_pool2d(out, 2)

class ImprovedCNN(nn.Module):
    """
    An improved CNN architecture with residual connections and better regularization.
    Configurable for different model sizes and tasks.
    """
    def __init__(self, config=None):
        super(ImprovedCNN, self).__init__()
        
        if config is None:
            config = {
                'input_channels': 3,
                'num_classes': 10,
                'model_size': 'medium',  # small, medium, large
                'dropout_rate': 0.5
            }
        
        # Define model size configurations
        size_configs = {
            'small': [32, 64, 128],
            'medium': [64, 128, 256],
            'large': [128, 256, 512]
        }
        
        channels = size_configs[config['model_size']]
        
        self.conv_block1 = ConvBlock(config['input_channels'], channels[0])
        self.conv_block2 = ConvBlock(channels[0], channels[1])
        self.conv_block3 = ConvBlock(channels[1], channels[2])
        
        # Adaptive pooling for flexible input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2] * 4 * 4, channels[2]),
            nn.ReLU(),
            nn.BatchNorm1d(channels[2]),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(channels[2], config['num_classes'])
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
