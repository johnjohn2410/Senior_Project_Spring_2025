"""
This script prepares the dataset for training a U-Net model on skin lesion segmentation. It first loads images and masks from
the ISIC 2016 dataset, ensuring they are correctly paired and free of corruption. The dataset is then split into training and
validation sets using an 80-20 split. MONAI transformations are applied to preprocess and augment the images, including
resizing, intensity scaling, random rotations, flips, and zooms. The dataset is structured into a MONAI CacheDataset to
speed up data loading. Finally, PyTorch DataLoaders are created for efficient batch processing, and a debugging step is
included to verify the dataset integrity before training begins.
"""

import os
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd,
    RandRotated, RandFlipd, RandZoomd, ToTensord, Compose
)
from PIL import Image
from sklearn.model_selection import train_test_split

# Defines dataset paths
LESION_IMAGE_DIR = "data/images/ISIC_2016_task1_training_data_images/"
LESION_MASK_DIR = "data/masks/ISIC_2016_task1_training_masks/"

# Loads images and masks files
lesion_image_list = sorted([
    os.path.join(LESION_IMAGE_DIR, img) for img in os.listdir(LESION_IMAGE_DIR) if img.endswith(".jpg")
])
lesion_mask_list = sorted([
    os.path.join(LESION_MASK_DIR, mask) for mask in os.listdir(LESION_MASK_DIR) if mask.endswith(".png")
])

# this ensures images and masks are correctly paried
assert len(lesion_image_list) == len(lesion_mask_list), \
    f"❌ ERROR: Mismatch between images ({len(lesion_image_list)}) and masks ({len(lesion_mask_list)})!"

# Debugs by finding faulty images
def verify_images(image_list):
    for img_path in image_list:
        try:
            with Image.open(img_path) as img:
                img.verify()  # Ensure the image is valid
        except Exception as e:
            print(f"⚠️ Skipping corrupt image: {img_path} - {e}")
            image_list.remove(img_path)  # Remove unreadable files
    return image_list

lesion_image_list = verify_images(lesion_image_list)
lesion_mask_list = verify_images(lesion_mask_list)

# 🔀 **Split into Training & Validation Sets**
train_images, val_images, train_masks, val_masks = train_test_split(
    lesion_image_list, lesion_mask_list, test_size=0.2, random_state=42
)

# MONAI preprocessing & augmentation pipeline**
lesion_transforms = Compose([
    LoadImaged(keys=["image", "mask"], reader="PILReader"),  # ✅ Load using PILReader
    EnsureChannelFirstd(keys=["image", "mask"]),  # ✅ Ensure correct channel format (grayscale -> 1 channel)
    Resized(keys=["image", "mask"], spatial_size=(256, 256)),  # ✅ Resize both images & masks
    ScaleIntensityd(keys=["image"]),  # ✅ Normalize only images (not masks)
    RandRotated(keys=["image", "mask"], range_x=15, prob=0.5),  # ✅ Random rotation
    RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),  # ✅ Horizontal flip
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),  # ✅ Vertical flip
    RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.5),  # ✅ Random zoom
    ToTensord(keys=["image", "mask"])  # ✅ Convert both to PyTorch tensors
])

# Creates dataset dictionaries
training_data_records = [{"image": img, "mask": mask} for img, mask in zip(train_images, train_masks)]
validation_data_records = [{"image": img, "mask": mask} for img, mask in zip(val_images, val_masks)]

# Defines MONAI CacheDataset**
training_dataset = CacheDataset(data=training_data_records, transform=lesion_transforms, cache_rate=1.0)
validation_dataset = CacheDataset(data=validation_data_records, transform=lesion_transforms, cache_rate=1.0)

# Creats dataloaders
training_data_loader = DataLoader(training_dataset, batch_size=8, shuffle=True, num_workers=0)  # ✅ `num_workers=0` for MacOS fix
validation_data_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=0)

# Prints dataset info
if __name__ == "__main__":
    print("\n🔍 Checking first 5 images and masks:")
    for i in range(min(5, len(train_images))):
        print(f"🖼 Image: {train_images[i]} - Exists: {os.path.exists(train_images[i])}")
        print(f"🎭 Mask: {train_masks[i]} - Exists: {os.path.exists(train_masks[i])}")

    # Load only the first batch to verify
    try:
        first_batch = next(iter(training_data_loader))
        print("\n✅ First sample loaded successfully:")
        print(f"🖼 Image Tensor Shape: {first_batch['image'].shape}")  # Expect [1, 256, 256]
        print(f"🎭 Mask Tensor Shape: {first_batch['mask'].shape}")  # Expect [1, 256, 256]
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
