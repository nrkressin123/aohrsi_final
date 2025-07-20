# This code has been adapted from the GitHub repository 'ck1972. (2025). "Semantic Segmentation
# of Aerial Imagery for Building Footprint Extraction Using DeepLabv3+ and TorchGeo" https://github.com/ck1972/University-GeoAI/blob/main/Mod2_Lab3e_TorchGeo_Building_Segmentation_deeplabv3_resnet101_Chit_GitHub1.ipynb

# --------------------------
# Import required libraries
# --------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio import features
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet101
from buildingregulariser import regularize_geodataframe
from sklearn.metrics import f1_score, jaccard_score
import scipy.ndimage as nd

# ---------------------------------------------------
# Load raster image and building polygons
# ---------------------------------------------------
img_path = 'data/basel_img1.tif'
with rasterio.open(img_path) as src:
    transform = src.transform
    height, width = src.height, src.width
    raster_crs = src.crs
    image = src.read()

gdf = gpd.read_file('data/building_footprints_img1.json')

# ---------------------------------------------------
# Rasterize building polygons into binary mask
# ---------------------------------------------------
mask = features.rasterize(
    ((geom, 1) for geom in gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# ---------------------------------------------------
# Define custom PyTorch dataset class
# ---------------------------------------------------
class AerialBuildingDataset(Dataset):
    def __init__(self, image, mask, patch_size=256):
        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.bands, self.height, self.width = image.shape
        self.patch_coords = [
            (i, j)
            for i in range(0, self.height - self.patch_size + 1, self.patch_size)
            for j in range(0, self.width - self.patch_size + 1, self.patch_size)
        ]

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        i, j = self.patch_coords[idx]
        img_patch = self.image[:, i:i+self.patch_size, j:j+self.patch_size]
        mask_patch = self.mask[i:i+self.patch_size, j:j+self.patch_size]
        return torch.tensor(img_patch, dtype=torch.float32), torch.tensor(mask_patch, dtype=torch.long)

# ---------------------------------------------------
# Create datasets and data loaders
# ---------------------------------------------------
dataset = AerialBuildingDataset(image, mask)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ---------------------------------------------------
# Visualize sample image and mask patch
# ---------------------------------------------------
def visualize_sample(dataset, index=0):
    img_patch, mask_patch = dataset[index]
    rgb = img_patch[:3].permute(1, 2, 0).numpy().astype(np.uint8)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Image Patch")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_patch, cmap='gray')
    plt.title("Mask Patch")
    plt.show()

visualize_sample(dataset, 350)

# ---------------------------------------------------
# Initialize DeepLabV3+ Model
# ---------------------------------------------------
print("initializing model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device loaded")
model = deeplabv3_resnet101(weights=None, num_classes=2).to(device)
print("model loaded")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print("optimizer loaded")
criterion = torch.nn.CrossEntropyLoss()
print("criterion loaded. proceeding to training")

# ---------------------------------------------------
# Training loop with early stopping & plots
# ---------------------------------------------------
from copy import deepcopy

num_epochs = 20
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_model = None

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for imgs, masks in train_loader:
        print('training')
        imgs, masks = imgs.to(device), masks.to(device)
        print(f"Using device: {imgs.device}")
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # --- Validation ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            print('eval')
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == masks).sum().item()
            total += masks.numel()

    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(correct / total)

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]*100:.2f}%, Val Acc: {val_accuracies[-1]*100:.2f}%")

    # --- Early Stopping ---
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model = deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

torch.save(best_model, "deeplabv3_building_segmentation.pth")

# Load the best model weights
model.load_state_dict(best_model)

# --- Plotting ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

# ------------------
# Model Evaluation
# ------------------
def compute_accuracy(outputs, masks):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / total

model.eval()
total_accuracy, total_iou = 0.0, 0.0
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)['out']
        total_accuracy += compute_accuracy(outputs, masks)
        preds = torch.argmax(outputs, dim=1)
        intersection = ((preds == 1) & (masks == 1)).sum().item()
        union = ((preds == 1) | (masks == 1)).sum().item()
        total_iou += intersection / union if union != 0 else 0

avg_acc = total_accuracy / len(test_loader)
avg_iou = total_iou / len(test_loader)
print(f"Test Accuracy: {avg_acc:.4f}, Test IoU: {avg_iou:.4f}")

# ------------------
# Dice and Boundary IoU
# ------------------
def dice_coefficient(preds, masks):
    preds = preds.view(-1)
    masks = masks.view(-1)
    intersection = ((preds == 1) & (masks == 1)).sum().item()
    pred_sum = (preds == 1).sum().item()
    mask_sum = (masks == 1).sum().item()
    return 2.0 * intersection / (pred_sum + mask_sum + 1e-7)

def boundary_iou(preds, masks, dilation_radius=1):
    preds_np = preds.cpu().numpy().astype(np.bool_)
    masks_np = masks.cpu().numpy().astype(np.bool_)
    struct = nd.generate_binary_structure(2, 1)
    boundary_pred = np.logical_xor(preds_np, nd.binary_erosion(preds_np, structure=struct))
    boundary_mask = np.logical_xor(masks_np, nd.binary_erosion(masks_np, structure=struct))
    boundary_pred = nd.binary_dilation(boundary_pred, structure=struct, iterations=dilation_radius)
    boundary_mask = nd.binary_dilation(boundary_mask, structure=struct, iterations=dilation_radius)
    intersection = np.logical_and(boundary_pred, boundary_mask).sum()
    union = np.logical_or(boundary_pred, boundary_mask).sum()
    return intersection / (union + 1e-7)

total_dice, total_biou = 0.0, 0.0
model.eval()
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)['out']
        preds = torch.argmax(outputs, dim=1)
        total_dice += dice_coefficient(preds, masks)
        total_biou += boundary_iou(preds[0], masks[0])

print(f"Average Dice Coefficient: {total_dice / len(test_loader):.4f}")
print(f"Average Boundary IoU: {total_biou / len(test_loader):.4f}")

# ------------------
# Full Image Inference
# ------------------
patch_size = 256
full_pred_mask = np.zeros((height, width), dtype=np.uint8)
model.eval()
with torch.no_grad():
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(patch_tensor)['out']
            pred_patch = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            full_pred_mask[i:i+patch_size, j:j+patch_size] = pred_patch

print("Inference completed on all patches.")

# ------------------
# Polygon Extraction
# ------------------
results = (
    {'properties': {'raster_val': v}, 'geometry': s}
    for s, v in features.shapes(full_pred_mask.astype(np.int16), transform=transform)
    if v == 1
)

geoms = list(results)
gdf_pred = gpd.GeoDataFrame.from_features(geoms, crs=raster_crs)
geojson_save_path = 'data/predicted_blds.geojson'
gdf_pred.to_file(geojson_save_path, driver='GeoJSON')
print(f"Predicted building footprints saved to {geojson_save_path}")

# ------------------
# Regularize Polygons
# ------------------
gdf_pred = regularize_geodataframe(gdf_pred)
gdf_pred.to_file(geojson_save_path.replace('.geojson', '_regularized.geojson'), driver='GeoJSON')
print("Regularized polygons saved.")

# Set number of samples to display
num_samples = 3

# Create a figure with 3 columns (RGB, True Mask, Predicted Mask) and `num_samples` rows
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

# Loop over selected sample indices
for idx in range(num_samples):
    # Extract patch coordinates from the dataset
    i, j = dataset.patch_coords[idx]

    # Extract RGB patch (first 3 bands) and transpose to (H, W, C) for visualization
    rgb_patch = image[:3, i:i+patch_size, j:j+patch_size].transpose(1, 2, 0)

    # Extract ground truth mask patch
    true_mask = mask[i:i+patch_size, j:j+patch_size]

    # Extract predicted mask patch
    pred_mask = full_pred_mask[i:i+patch_size, j:j+patch_size]

    # Plot RGB image
    axes[idx, 0].imshow(rgb_patch.astype(np.uint8))
    axes[idx, 0].set_title("RGB Image")
    axes[idx, 0].axis("off")

    # Plot ground truth mask
    axes[idx, 1].imshow(true_mask, cmap='gray')
    axes[idx, 1].set_title("Ground Truth Mask")
    axes[idx, 1].axis("off")

    # Plot predicted mask
    axes[idx, 2].imshow(pred_mask, cmap='gray')
    axes[idx, 2].set_title("Predicted Mask")
    axes[idx, 2].axis("off")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# ------------------
# Save Trained Model
# ------------------

# Define the path to save the model weights in Google Drive
model_save_path = "models/deeplabv3_resnet101_model.pth"

# Save the model's learned parameters (state_dict)
torch.save(model.state_dict(), model_save_path)

# Print confirmation
print(f"Trained model saved to {model_save_path}")

# ------------------
# Automated Report Generation
# ------------------
report_path = "models/segmentation_report.txt"
with open(report_path, "w") as report:
    report.write("Semantic Segmentation Report\n")
    report.write("============================\n")
    report.write(f"Model: DeepLabV3+ (ResNet101)\n")
    report.write(f"Epochs Trained: {num_epochs}\n")
    report.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
    report.write(f"Test Accuracy: {avg_acc:.4f}\n")
    report.write(f"Test IoU: {avg_iou:.4f}\n")
    report.write(f"Dice Coefficient: {total_dice / len(test_loader):.4f}\n")
    report.write(f"Boundary IoU: {total_biou / len(test_loader):.4f}\n")
    report.write(f"Original Image Size: {image.shape[1]} x {image.shape[2]}\n")
    report.write(f"Patch Size: {patch_size} x {patch_size}\n")
    report.write("\nPredicted footprints saved at:\n")
    report.write(f"{geojson_save_path}\n")
    report.write("Regularized polygons saved at:\n")
    report.write(f"{geojson_save_path.replace('.geojson', '_regularized.geojson')}\n")

print(f"Report generated and saved to {report_path}")