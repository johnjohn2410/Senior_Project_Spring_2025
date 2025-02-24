"""
This script trains a U-Net model for skin lesion segmentation using the ISIC 2016 dataset. It first loads and preprocesses
the dataset, ensuring images and masks are correctly paired and transformed into grayscale for consistency. The dataset is
wrapped into a PyTorch DataLoader for batch processing. The U-Net model is then initialized, configured with a loss function
(Dice Loss), and optimized using Adam. Before training begins, a batch of images and masks is visualized to verify data
integrity. The training loop runs for 20 epochs, where the model processes images, computes loss, and updates weights using
backpropagation. The script outputs loss values per epoch and saves the trained model to disk for later use.
"""

import os
import torch
import monai
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, RandRotate, RandFlip, RandZoom,
    Compose, Resize, ToTensor, AsDiscrete
)
from torch.optim import Adam

# Allows multiprocessing on Windows and macOS
if __name__ == "__main__":

    # Define dataset paths
    LESION_IMAGE_DIR = "data/images/ISIC_2016_task1_training_data_images/"
    LESION_MASK_DIR = "data/masks/ISIC_2016_task1_training_masks/"
    TRAINED_MODEL_PATH = "models/skin_lesion_segmentation.pth"

    # 📂 **Load training images and masks**
    lesion_image_list = sorted(
        [os.path.join(LESION_IMAGE_DIR, img) for img in os.listdir(LESION_IMAGE_DIR) if img.endswith(".jpg")]
    )
    lesion_mask_list = sorted(
        [os.path.join(LESION_MASK_DIR, mask) for mask in os.listdir(LESION_MASK_DIR) if mask.endswith("_Segmentation.png")]
    )

    # Check if dataset exists
    if not lesion_image_list or not lesion_mask_list:
        raise FileNotFoundError(f"❌ ERROR: No images or masks found in {LESION_IMAGE_DIR} or {LESION_MASK_DIR}")

    # Check if image/mask counts match**
    if len(lesion_image_list) != len(lesion_mask_list):
        raise ValueError(f"❌ ERROR: Mismatch between images ({len(lesion_image_list)}) and masks ({len(lesion_mask_list)})!")

    # Debugs by printing the first 5 image/mask paths
    print("\n🔍 Checking first 5 training image paths:")
    for img_path in lesion_image_list[:5]:
        print(f"🖼 Image: {img_path} - Exists: {os.path.exists(img_path)}")

    print("\n🔍 Checking first 5 training mask paths:")
    for mask_path in lesion_mask_list[:5]:
        print(f"🎭 Mask: {mask_path} - Exists: {os.path.exists(mask_path)}")

    # Define Image Transformations (Convert RGB → Grayscale)
    image_transforms = Compose([
        LoadImage(reader="PILReader", image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((256, 256)),
        ToTensor(),
        lambda x: x.mean(dim=0, keepdim=True)  # 🔥 Convert RGB to Grayscale (3 → 1 channel)
    ])

    # Mask Transformations
    mask_transforms = Compose([
        LoadImage(reader="PILReader", image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((256, 256)),
        AsDiscrete(threshold=0.5),  # Ensure mask is binary
        ToTensor()
    ])

    # Create dataset
    class LesionDataset(Dataset):
        def __init__(self, images, masks, image_transforms, mask_transforms):
            self.images = images
            self.masks = masks
            self.image_transforms = image_transforms
            self.mask_transforms = mask_transforms

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.image_transforms(self.images[idx])
            mask = self.mask_transforms(self.masks[idx])
            return {"image": image, "label": mask}  # ✅ Using "label" as key

    lesion_dataset = LesionDataset(
        lesion_image_list, lesion_mask_list, image_transforms, mask_transforms
    )

    # Cross-Platform DataLoader (MacOS vs. Windows/Linux)
    num_workers = 0 if os.name == "posix" else 2  # `posix` = macOS/Linux, Windows needs 2+

    lesion_data_loader = DataLoader(lesion_dataset, batch_size=8, shuffle=True, num_workers=num_workers)

    # Define UNet model
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skin_lesion_model = UNet(
        spatial_dims=2,
        in_channels=1,  # Grayscale images
        out_channels=1,  # Binary masks
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(computation_device)

    # Define loss function & optimizer
    lesion_loss_function = DiceLoss(sigmoid=True)
    lesion_optimizer = Adam(skin_lesion_model.parameters(), lr=1e-4)
    dice_score_metric = DiceMetric(include_background=False, reduction="mean")

    # Checks a Batch Before Training
    data_batch = next(iter(lesion_data_loader))
    images, masks = data_batch["image"], data_batch["label"]
    print(f"\n🖼 Image Batch Shape: {images.shape}")
    print(f"🎭 Mask Batch Shape: {masks.shape}")

    # Visualize Sample Batch
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        axes[0, i].imshow(images[i].squeeze(0).cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(masks[i].squeeze(0).cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Images")
    axes[1, 0].set_title("Masks")
    plt.show()

    # Training loop
    num_training_epochs = 20
    for epoch_index in range(num_training_epochs):
        skin_lesion_model.train()
        cumulative_epoch_loss = 0

        for data_batch in lesion_data_loader:
            lesion_optimizer.zero_grad()
            input_images = data_batch["image"].to(computation_device)
            ground_truth_masks = data_batch["label"].to(computation_device)  # ✅ Use "label"

            lesion_predictions = skin_lesion_model(input_images)
            training_loss = lesion_loss_function(lesion_predictions, ground_truth_masks)
            training_loss.backward()
            lesion_optimizer.step()
            cumulative_epoch_loss += training_loss.item()

        print(f"📈 Epoch {epoch_index + 1}/{num_training_epochs}, Loss: {cumulative_epoch_loss:.4f}")

    # Save trained model
    torch.save(skin_lesion_model.state_dict(), TRAINED_MODEL_PATH)
    print("✅ Training complete. Model saved at:", TRAINED_MODEL_PATH)
