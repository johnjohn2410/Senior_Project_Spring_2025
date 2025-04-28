# train_vm_unet.py (Transformer-based UNetV2)
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from dataset import get_dataloaders
from unet_v2.unet_v2 import UNetV2

wandb.init(project="skin-lesion", name="unetv2-transformer-300ep", config={
    "epochs": 300,
    "batch_size": 80,
    "lr": 1e-4,
    "model": "UNetV2"
})

train_loader, val_loader = get_dataloaders(batch_size=wandb.config["batch_size"])

model = UNetV2(n_classes=2, deep_supervision=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load pretrained weights for Transformer model
pretrained_path = "./pretrained/polyp_pvt/PolypPVT.pth"
if os.path.exists(pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
else:
    print(f"Pretrained weights not found at {pretrained_path}. Continuing without pretrained weights.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config["lr"])

# Resume from checkpoint if available
checkpoint_path = "checkpoints/transformer_latest.pth"
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from checkpoint at epoch {start_epoch}")

for epoch in range(start_epoch, wandb.config["epochs"]):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].squeeze(1).to(device).long().clamp(max=1)

        outputs = model(images)
        loss = sum([
            criterion(
                nn.functional.interpolate(output, size=masks.shape[-2:], mode='bilinear', align_corners=False),
                masks
            ) for output in outputs]) / len(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    wandb.log({"epoch": epoch+1, "train_loss": avg_loss})
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].squeeze(1).to(device).long().clamp(max=1)
            outputs = model(images)
            loss = sum([
                criterion(
                    nn.functional.interpolate(output, size=masks.shape[-2:], mode='bilinear', align_corners=False),
                    masks
                ) for output in outputs]) / len(outputs)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

wandb.finish()