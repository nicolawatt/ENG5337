# imports and setup
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
try:
    from tqdm import tqdm, trange
except ImportError:
    def tqdm(iterable, desc=None, leave=None, **kwargs):
        return iterable
    def trange(*args, desc=None, **kwargs):
        return range(*args)
 
# device
if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
 
# config — paths relative to script dir (works on any machine)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "dataset"
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "val"
 
IMAGE_SIZE = 128
NUM_CLASSES = None   # inferred from dataset (train folder subdirs)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-5          # lower LR helps with imbalanced data
EARLY_STOP_PATIENCE = 10     # give minority classes more time to learn
LABEL_SMOOTHING = 0.1        # reduces overconfidence, helps generalisation
BEST_MODEL_PATH = SCRIPT_DIR / "best_mri_cnn.pt"
CONFUSION_FIG_PATH = SCRIPT_DIR / "confusion_matrix.png"
 
if __name__ == '__main__':
    train_dir_str = str(TRAIN_DIR)
    valid_dir_str = str(VALID_DIR)
    if not TRAIN_DIR.is_dir():
        print(f"Error: TRAIN_DIR not found: {TRAIN_DIR}")
        print("Ensure dataset/train and dataset/valid exist (e.g. run autolabel.py and copy to dataset, or create dataset manually).")
        exit(1)
    if not VALID_DIR.is_dir():
        print(f"Error: VALID_DIR not found: {VALID_DIR}")
        exit(1)
 
    stats_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
 
    stats_dataset = datasets.ImageFolder(root=train_dir_str, transform=stats_transform)
    stats_loader = DataLoader(stats_dataset, batch_size=64, shuffle=False, num_workers=0)
 
    #calculate mean and std
    pixel_sum = 0
    pixel_squared_sum = 0
    num_pixels = 0
 
    for images, _ in tqdm(stats_loader, desc="Calculating mean and std"):
        pixel_sum += images.sum()
        pixel_squared_sum += (images ** 2).sum()
        num_pixels += images.numel()
 
    train_mean = pixel_sum / num_pixels
    train_std = torch.sqrt(torch.clamp(
        (pixel_squared_sum / num_pixels) - (train_mean ** 2), min=1e-8))
 
    print(f"Training set mean: {train_mean:.4f}")
    print(f"Training set std:  {train_std:.4f}")
 
    # train: optional horizontal flip for left/right robustness
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std]),
    ])
 
    valid_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std]),
    ])
 
    train_dataset = datasets.ImageFolder(root=train_dir_str, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir_str, transform=valid_transform)
 
    # Class distribution and weights for imbalanced data
    targets = np.array(train_dataset.targets)
    class_names = train_dataset.classes
    num_classes = len(class_names)  # use actual number of classes from dataset
    unique, counts = np.unique(targets, return_counts=True)
    class_counts = dict(zip(unique, counts))
    n_per_class = np.array([class_counts.get(i, 0) for i in range(len(class_names))], dtype=np.float64)
    n_per_class = np.maximum(n_per_class, 1.0)  # avoid div by zero
    # Inverse frequency weights (normalized so mean=1)
    class_weights = 1.0 / n_per_class
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
 
    # Per-sample weights for balanced sampling (each sample weight = 1/count of its class)
    sample_weights = np.array([1.0 / n_per_class[t] for t in targets], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_dataset)}, Valid images: {len(valid_dataset)}")
    print("Per-class train counts:", {class_names[i]: int(n_per_class[i]) for i in range(len(class_names))})
    print("Class weights (for loss):", {class_names[i]: f"{class_weights[i]:.3f}" for i in range(len(class_names))})
    print()
 
 
    class MRI_CNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
 
            # block 1
            self.conv1a = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, padding=1)
            self.bn1a = nn.BatchNorm2d(32)
            self.conv1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            self.bn1b = nn.BatchNorm2d(32)
            self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)
 
            # block 2
            self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.bn2a = nn.BatchNorm2d(64)
            self.conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            self.bn2b = nn.BatchNorm2d(64)
            self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)
 
            # block 3
            self.conv3a = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
            self.bn3a = nn.BatchNorm2d(128)
            self.conv3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            self.bn3b = nn.BatchNorm2d(128)
            self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
 
            # classifier
            self.fc1     = nn.Linear(128 * 16 * 16, 256)
            self.dropout = nn.Dropout(p=0.5)
            self.fc2     = nn.Linear(256, num_classes)
 
        def forward(self, x):
            # block 1
            x = F.relu(self.bn1a(self.conv1a(x)))
            x = F.relu(self.bn1b(self.conv1b(x)))
            x = self.pool1(x)
 
            # block 2
            x = F.relu(self.bn2a(self.conv2a(x)))
            x = F.relu(self.bn2b(self.conv2b(x)))
            x = self.pool2(x)
 
            # block 3
            x = F.relu(self.bn3a(self.conv3a(x)))
            x = F.relu(self.bn3b(self.conv3b(x)))
            x = self.pool3(x)
 
            # flatten
            x = x.view(x.size(0), -1)
 
            # classifier
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
 
            return x
 
 
    model = MRI_CNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")
 
 
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.3)
 
    class EarlyStopping:
        def __init__(self, patience=EARLY_STOP_PATIENCE, path=BEST_MODEL_PATH):
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
 
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, path=BEST_MODEL_PATH)
 
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
 
    # load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
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
 
    # confusion matrix
    classes = class_names
    n_classes = len(classes)
 
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[true][pred] += 1
 
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap='Blues')
 
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Validation Set')
 
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center',
                    color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
 
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(CONFUSION_FIG_PATH, dpi=150)
    print(f"Confusion matrix saved to: {CONFUSION_FIG_PATH}")
    plt.show()
 
 