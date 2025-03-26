"""
This script loads a trained U-Net model to perform segmentation on test images and visualize the results. It first loads
predefined test images from a directory, applies preprocessing transformations, and prepares them for model inference.
The script then loads the trained segmentation model and applies it to the test images using a sliding window approach.
The predicted segmentation masks are extracted and displayed alongside the original images for comparison. This allows us
to evaluate the model's performance on unseen data and validate how well it generalizes to new skin lesion images.
"""

import platform
import multiprocessing
if platform.system() == "Darwin":  # macOS
    multiprocessing.set_start_method("fork", force=True)
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor, Compose
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader
from image_tools import remove_noise_and_extract_largest  # Remove sharpen_image import


# Define paths for test images and trained model
TEST_IMAGE_DIRECTORY = "skin-lesion-segmentation/data/test_images/ISIC_2016_task1_training_images_tests/"
TEST_MASK_DIRECTORY = "skin-lesion-segmentation/data/test_masks/ISIC_2016_task1_training_masks_tests/"
MODEL_FILE_PATH = "skin-lesion-segmentation/models/skin_lesion_segmentation.pth"

# Check if image file exists
if not os.path.exists(TEST_IMAGE_DIRECTORY):
    raise FileNotFoundError(f"❌ ERROR: Image file not found at {TEST_IMAGE_DIRECTORY}")

# Check if model file exists
if not os.path.exists(MODEL_FILE_PATH):
    raise FileNotFoundError(f"❌ ERROR: Model file not found at {MODEL_FILE_PATH}")

# Load test images
test_image_list = sorted(
    [os.path.join(TEST_IMAGE_DIRECTORY, img) for img in os.listdir(TEST_IMAGE_DIRECTORY) if img.endswith(".jpg")])

# Load true masks
test_mask_list = sorted(
    [os.path.join(TEST_MASK_DIRECTORY, mask) for mask in os.listdir(TEST_MASK_DIRECTORY) if mask.endswith(".png")])

if len(test_image_list) != len(test_mask_list):
    raise ValueError(f"❌ ERROR: Mismatch between test images ({len(test_image_list)}) and masks ({len(test_mask_list)})!")

# Define transformation function to replace lambda
def convert_to_grayscale(image):
    return image.mean(dim=0, keepdim=True)  # Convert RGB to Grayscale (3 → 1 channel)

# Define MONAI transformations
test_image_transforms = Compose([
    LoadImage(reader="PILReader", image_only=True),
    EnsureChannelFirst(),
    Resize((256, 256)),
    ScaleIntensity(),
    ToTensor(),
    convert_to_grayscale  # Use named function instead of lambda
])

# Create dataset
class LesionDataset(Dataset):
    def __init__(self, images, image_transforms):
        self.images = images
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.image_transforms(self.images[idx])
        return {"image": image}

# Create dataset and DataLoader
test_data_samples = LesionDataset(test_image_list, test_image_transforms)
test_data_loader = DataLoader(test_data_samples, batch_size=1, shuffle=False, num_workers=2)

# Load trained model
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device_type)

segmentation_model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device_type))
segmentation_model.eval()

# Prediction function
def generate_segmentation_masks(model, dataloader):
    mask_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            image_tensor = batch["image"].to(device_type)
            predicted_mask = sliding_window_inference(image_tensor, (256, 256), 4, model)
            predicted_mask = torch.sigmoid(predicted_mask).cpu().numpy()
            mask_predictions.append(predicted_mask)
    return mask_predictions


if __name__ == "__main__":
    print("Starting segmentation...")

    # Create dataset and DataLoader inside __main__
    test_data_samples = LesionDataset(test_image_list, test_image_transforms)
    test_data_loader = DataLoader(test_data_samples, batch_size=1, shuffle=False, num_workers=2)

    # Run predictions
    test_image_masks = generate_segmentation_masks(segmentation_model, test_data_loader)

    # Define threshold
    threshold = 0.6

    # Clean masks
    cleaned_masks = [
        remove_noise_and_extract_largest(predicted_mask[0][0], threshold=threshold)
        for predicted_mask in test_image_masks
    ]

    # Visualize a few results
    sample_count = min(5, len(test_image_list))
    figure, axis_matrix = plt.subplots(sample_count, 4, figsize=(10, 12.5))  # Adjusted for 4 columns

    # Reduce spacing between subplots
    plt.subplots_adjust(wspace = 0.2, hspace = 0.2)

    for index in range(sample_count):
        input_image = Image.open(test_image_list[index])  # Load and convert to grayscale
        true_mask = Image.open(test_mask_list[index]).convert("L")  # Load true mask
        predicted_mask = test_image_masks[index][0][0]  # First batch, first channel
        cleaned_mask = cleaned_masks[index]  # Use precomputed cleaned mask

        # Display original image
        axis_matrix[index, 0].imshow(input_image)
        axis_matrix[index, 0].set_title("Original Image")
        axis_matrix[index, 0].axis("off")

        # Display true mask
        axis_matrix[index, 1].imshow(true_mask, cmap='jet')
        axis_matrix[index, 1].set_title("True Mask")
        axis_matrix[index, 1].axis("off")

        # Display predicted mask
        axis_matrix[index, 2].imshow(predicted_mask, cmap='jet')
        axis_matrix[index, 2].set_title("Predicted Mask")
        axis_matrix[index, 2].axis("off")

        # Display cleaned mask
        axis_matrix[index, 3].imshow(cleaned_mask, cmap='jet')
        axis_matrix[index, 3].set_title(f"Cleaned Mask (T={threshold})")
        axis_matrix[index, 3].axis("off")

    plt.tight_layout()
    plt.show()

    print("Segmentation complete! Displaying true masks, predicted masks, and cleaned masks.")
