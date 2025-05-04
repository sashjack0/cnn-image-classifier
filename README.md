# 🧠 CNN Image Classifier

> A production-ready deep learning project built with PyTorch for image classification tasks. Features a modern CNN architecture with residual connections, advanced data augmentation, and comprehensive training pipeline.

![PyTorch](https://img.shields.io/badge/built%20with-PyTorch-ff5050?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-Active-green)

## 🌟 Features

- ✨ **Modern Architecture**
  - Custom CNN with residual connections
  - Configurable model sizes (small/medium/large)
  - Batch normalization and dropout for regularization

- 🚀 **Advanced Training Pipeline**
  - Automatic Mixed Precision (AMP) training
  - One Cycle Learning Rate scheduling
  - Early stopping and model checkpointing
  - Training curve visualization

- 📊 **Data Management**
  - Support for CIFAR-10/100 datasets
  - Advanced data augmentation pipeline
  - Proper train/validation/test splitting
  - Efficient data loading with proper normalization

- 🛠 **Developer Friendly**
  - Clean, modular code structure
  - Configuration-based training
  - Comprehensive documentation
  - Ready for experimentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sashjack0/cnn-image-classifier.git
cd cnn-image-classifier
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cnn-env python=3.8
conda activate cnn-env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

1. Review and modify the configuration in `configs/default_config.json` if needed.

2. Start training:
```bash
python training/train.py
```

3. Monitor training progress:
- Training curves will be saved in `checkpoints/training_curves_[timestamp].png`
- Model checkpoints will be saved in `checkpoints/best_model_[timestamp].pth`
- Training configuration and results will be saved as JSON files

## 📁 Project Structure

```
cnn-image-classifier/
├── configs/
│   └── default_config.json     # Training configuration
├── datasets/
│   └── dataset.py             # Dataset and data loading utilities
├── models/
│   └── cnn_model.py          # Model architecture definitions
├── training/
│   └── train.py              # Training script
├── utils/                    # Utility functions
├── notebooks/                # Jupyter notebooks for experimentation
├── tests/                    # Unit tests
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## 🔧 Configuration

The project uses a configuration-based approach for easy experimentation. Key configuration sections:

- **Training**: Epochs, batch size, learning rate, etc.
- **Model**: Architecture settings, model size, dropout rate
- **Data**: Dataset choice, data directory, validation split
- **Logging**: Checkpoint directory, logging frequency

See `configs/default_config.json` for all available options.

## 📈 Performance

Default configuration achieves:
- CIFAR-10: ~93% test accuracy
- CIFAR-100: ~72% test accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sachin Bhandary**
- GitHub: [@sashjack0](https://github.com/sashjack0)

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CIFAR Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
