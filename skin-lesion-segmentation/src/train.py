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
import matplotlib.pyplot as plt
import wandb
import numpy as np
from torch.optim import Adam
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from dataset import get_dataloaders

DEBUG_VISUALIZE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "models/skin_lesion_segmentation.pth"
BEST_MODEL_PATH = "models/best_model.pth"

wandb.init(project="skin-lesion-segmentation", name="train-rgb-cradle-run")
wandb.config = {
    "epochs": 300,
    "batch_size": 8,
    "image_size": "256x256",
    "input_channels": 3,
    "model": "UNet",
    "optimizer": "Adam",
    "loss_function": "DiceLoss"
}

train_loader, val_loader = get_dataloaders(batch_size=wandb.config["batch_size"])

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

criterion = DiceLoss(sigmoid=True)
optimizer = Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

best_dice = 0.0

for epoch in range(wandb.config["epochs"]):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)

    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for val_batch in val_loader:
            val_images = val_batch["image"].to(device)
            val_masks = val_batch["mask"].to(device)

            val_preds = model(val_images)
            val_preds = torch.sigmoid(val_preds)
            val_preds = (val_preds > 0.3).float()

            dice_metric(y_pred=val_preds, y=val_masks)

    avg_val_dice = dice_metric.aggregate().item()

    print(f"📈 Epoch {epoch + 1}/{wandb.config['epochs']} - Loss: {avg_train_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_dice_score": avg_val_dice
    })

    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"💾 New best model saved with Dice: {best_dice:.4f}")

    if (epoch + 1) % 10 == 0 or epoch == 0:
        input_img = val_images[0].permute(1, 2, 0).cpu().numpy()
        true_mask = val_masks[0][0].cpu().numpy()
        pred_mask = val_preds[0][0].cpu().numpy()

        wandb.log({
            "Input Image": wandb.Image(input_img, caption="Input"),
            "True Mask": wandb.Image(true_mask, caption="Ground Truth"),
            "Predicted Mask": wandb.Image(pred_mask, caption="Prediction")
        })

        if DEBUG_VISUALIZE:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(input_img)
            axs[0].set_title("Input Image")
            axs[1].imshow(true_mask, cmap="gray")
            axs[1].set_title("Ground Truth Mask")
            axs[2].imshow(pred_mask, cmap="gray")
            axs[2].set_title("Predicted Mask")
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            plt.show()

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("✅ Training complete. Final model saved at:", MODEL_SAVE_PATH)

wandb.finish()
