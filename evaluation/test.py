import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import sys

# Add parent path to access modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.cnn_model import SimpleCNN  # Make sure model class matches
from datasets.dataset import get_data_loaders  # Reuse function if available

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Evaluation
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
accuracy = 100 * correct / total
print(f'\n🎯 Test Accuracy: {accuracy:.2f}%')

# Per-class accuracy
classes = test_dataset.classes
class_correct = [0.] * 10
class_total = [0.] * 10

for i in range(len(all_labels)):
    label = all_labels[i]
    pred = all_preds[i]
    if label == pred:
        class_correct[label] += 1
    class_total[label] += 1

print("\n📊 Per-Class Accuracy:")
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f'{classes[i]:<10}: {acc:.2f}%')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png")
plt.show()
