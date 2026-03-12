#imports and setup
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm, trange
except ImportError:
    def tqdm(iterable, desc=None, leave=None, **kwargs):
        return iterable
    def trange(*args, desc=None, **kwargs):
        return range(*args)
import os
 
#use mps (mac)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
 
#load and preprocess data
IMAGE_SIZE = 128 #original: 224
 
TRAIN_DIR = '/Users/nicolawatt/Documents/University - Glasgow/ENG5337/Environment/Group Project/train'
VALID_DIR = '/Users/nicolawatt/Documents/University - Glasgow/ENG5337/Environment/Group Project/valid'
 
if __name__ == '__main__':
    stats_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #convert to grayscale
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), #resize to 244x244
        transforms.ToTensor(), #convert to tensor
    ])
 
    stats_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=stats_transform)
    stats_loader  = DataLoader(stats_dataset, batch_size=64, shuffle=False, num_workers=0)  # Windows 下用 0 避免多进程报错
 
    #calculate mean and std
    pixel_sum = 0
    pixel_squared_sum = 0
    num_pixels = 0
 
    for images, _ in tqdm(stats_loader, desc="Calculating mean and std"):
        pixel_sum += images.sum()
        pixel_squared_sum += (images ** 2).sum()
        num_pixels += images.numel()
 
    train_mean = pixel_sum / num_pixels
    train_std = torch.sqrt((pixel_squared_sum / num_pixels) - (train_mean ** 2))
 
    print(f"Training set mean: {train_mean:.4f}")
    print(f"Training set std: {train_std:.4f}")
 
    #define transforms with normalisation (no flip/rotation)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #convert to grayscale
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), #resize to 224x224
        transforms.ToTensor(), #convert to tensor
        transforms.Normalize(mean=[train_mean], std=[train_std]) #normalise
    ])
 
    valid_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #convert to grayscale
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), #resize to 224x224
        transforms.ToTensor(), #convert to tensor
        transforms.Normalize(mean=[train_mean], std=[train_std]) #normalise
    ])
 
    #load datasets and dataloaders
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=VALID_DIR, transform=valid_transform)
 
    batch_size = 32
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
 
    #print(f"Classes: {train_dataset.classes}")
    #print(f"Training images: {len(train_dataset)}")
    #print(f"Validation images: {len(valid_dataset)}")
 
 
    class MRI_CNN(nn.Module):
       def __init__(self, num_classes=4):
          super().__init__()
 
          #block 1
          self.conv1a = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, padding=1)
          self.bn1a = nn.BatchNorm2d(32)
          self.conv1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
          self.bn1b = nn.BatchNorm2d(32)
          self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)
 
          #block 2
          self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
          self.bn2a = nn.BatchNorm2d(64)
          self.conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
          self.bn2b = nn.BatchNorm2d(64)
          self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)
 
          #block 3
          self.conv3a = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
          self.bn3a = nn.BatchNorm2d(128)
          self.conv3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
          self.bn3b = nn.BatchNorm2d(128)
          self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
 
          #classifier
          self.fc1     = nn.Linear(128 * 16 * 16, 256)
          self.dropout = nn.Dropout(p=0.5)
          self.fc2     = nn.Linear(256, num_classes)
          
       def forward(self, x):
          #block 1
          x = F.relu(self.bn1a(self.conv1a(x)))
          x = F.relu(self.bn1b(self.conv1b(x)))
          x = self.pool1(x)
 
          #block 2
          x = F.relu(self.bn2a(self.conv2a(x)))
          x = F.relu(self.bn2b(self.conv2b(x)))
          x = self.pool2(x)
 
          #block 3
          x = F.relu(self.bn3a(self.conv3a(x)))
          x = F.relu(self.bn3b(self.conv3b(x)))  
          x = self.pool3(x)
 
          #flatten
          x = x.view(x.size(0), -1)
 
          #classifier
          x = F.relu(self.fc1(x))
          x = self.dropout(x)
          x = self.fc2(x)
 
          return x
 
 
    #dummy forward pass
    model = MRI_CNN(num_classes=4).to(device)
 
    dummy = torch.zeros(4, 1, 128, 128).to(device)  # batch of 4, 1 channel, 128x128
    output = model(dummy)
 
    #print(f"Input shape  : {dummy.shape}")
    #print(f"Output shape : {output.shape}")   # expect (4, 4)
    #print()
 
    #count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total trainable parameters: {total_params:,}")
    #print()
    #print(model)
 
 
    #define loss
    criterion = nn.CrossEntropyLoss()
 
    #define optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
 
    #define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)
 
    class EarlyStopping:
       def __init__(self, patience=5, path='best_mri_cnn.pt'):
          self.patience = patience
          self.path = path
          self.best_acc = 0.0
          self.counter = 0
          self.early_stop = False
 
       def __call__(self, val_acc, model):
          if val_acc > self.best_acc:
             self.best_acc = val_acc
             self.counter = 0
             torch.save(model.state_dict(), self.path)
          else:
             self.counter += 1
             if self.counter >= self.patience:
                self.early_stop = True
 
    EPOCHS = 30
    early_stopping = EarlyStopping(patience=5, path='best_mri_cnn.pt')
 
    best_valid_acc = 0.0
    train_losses, valid_accs = [], []
 
    for epoch in trange(EPOCHS, desc="Training epochs"):
 
       #training
       model.train()
       running_loss = 0.0
 
       for images, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1} train", leave=False):
          images, labels = images.to(device), labels.to(device)
 
          optimiser.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimiser.step()
 
          running_loss += loss.item()
 
       avg_loss = running_loss / len(train_loader)
       train_losses.append(avg_loss)
 
       #validation
       model.eval()
       correct = 0
 
       with torch.no_grad():
          for images, labels in tqdm(valid_loader, desc=f"  Epoch {epoch+1} valid", leave=False):
             images, labels = images.to(device), labels.to(device)
             outputs = model(images)
             predictions = torch.argmax(outputs, dim=1)
             correct += (predictions == labels).sum().item()
 
       valid_acc = correct / len(valid_dataset)
       valid_accs.append(valid_acc)
 
       #step scheduler
       scheduler.step()
 
       #early stopping
       early_stopping(valid_acc, model)
 
       print(f"Epoch [{epoch+1}/{EPOCHS}]  "
             f"Loss: {avg_loss:.4f}  "
              f"Val Acc: {valid_acc:.4f}  "
              f"Best: {early_stopping.best_acc:.4f}  "
              f"Patience: {early_stopping.counter}/{early_stopping.patience}")
 
       if early_stopping.early_stop:
          print(f"\nEarly stopping triggered at epoch {epoch+1}")
          break
   
    print(f"\nBest validation accuracy: {early_stopping.best_acc:.4f}")
 
    #load best model
    model.load_state_dict(torch.load('./best_mri_cnn.pt', weights_only=True))
    model.eval()
 
    correct = 0
    all_preds  = []
    all_labels = []
 
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct    += (predictions == labels).sum().item()
 
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    valid_acc = correct / len(valid_dataset)
    print(f"Validation accuracy: {valid_acc:.4f}  ({correct}/{len(valid_dataset)})")
 
    #confusion matrix
    classes = train_dataset.classes
    n_classes = len(classes)
 
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[true][pred] += 1
 
    #plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_matrix, cmap='Blues')
 
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Validation Set')
 
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center',
                    color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
 
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
