# 🧠 CNN Image Classifier ![PyTorch](https://img.shields.io/badge/built%20with-PyTorch-ff5050?style=flat&logo=pytorch) ![Status](https://img.shields.io/badge/status-Active-blue)

> A modular deep learning project built using PyTorch that classifies images from datasets like CIFAR-10 using a custom CNN architecture. Clean code, structured folders, and ready for training, testing, and deployment.

---

## 📌 Features

- ✅ Custom-built CNN architecture using PyTorch  
- 📁 Clean modular structure for easy extension  
- 🧪 Ready for training, validation, and evaluation  
- 💾 Saves best-performing model checkpoints  
- 🚀 GPU acceleration with auto-detection  
- 🔬 Unit testing and configuration support (coming soon)  

---

## 📷 Sample Output (Optional Screenshot)

> _(Add a training plot or accuracy log screenshot here when available)_

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/sashjack0/cnn-image-classifier.git
cd cnn-image-classifier
```

### 2. Set Up Python Environment

Using Conda:

```bash
conda create -n cnn-env python=3.10
conda activate cnn-env
pip install -r requirements.txt
```

Or using venv:

```bash
python -m venv cnn-env
source cnn-env/bin/activate  # On Windows: cnn-env\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python training/train.py
```

---

## 🧱 Project Structure

```bash
cnn-image-classifier/
├── configs/               # Training configs (to be added)
├── datasets/              # Data loading and transformation scripts
│   └── dataset.py
├── models/                # Model architectures
│   └── cnn_model.py
├── training/              # Training and evaluation scripts
│   └── train.py
├── utils/                 # (Planned: helpers, metrics)
├── tests/                 # (Planned: unit tests)
├── notebooks/             # (Optional: for experimentation)
├── requirements.txt       # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Example Use Case

> This CNN classifier can be used in academic ML tasks, hackathons, or as a base for more complex computer vision models like ResNet, ViT, etc.

---

## 👨‍💻 Author

**Sachin Bhandary**  
_PyTorch & AI Engineer_  
GitHub: [@sashjack0](https://github.com/sashjack0)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE)

---

## 📌 Next Steps

- ✅ Add unit tests to `tests/`  
- 📊 Add training curves or logs to `notebooks/`  
- 🛠️ Add `test.py` and `metrics.py` for evaluation  
- 💻 Add model export (`.pth`) + loading script  
