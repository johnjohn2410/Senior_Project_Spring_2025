# train_unetv2_mamba.py (Mamba-based UNetV2)
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataset import get_dataloaders
from vm_unet_v2.VM_UNetV2 import VMUNetV2

wandb.init(project="skin-lesion", name="unetv2-mamba-test", config={
    "epochs": 2,
    "batch_size": 2,
    "lr": 1e-4,
    "model": "VMUNetV2"
})

train_loader, val_loader = get_dataloaders(batch_size=wandb.config["batch_size"])

model = VMUNetV2(n_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=wandb.config["lr"])

for epoch in range(wandb.config["epochs"]):
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
    wandb.log({"epoch": epoch+1, "loss": avg_loss})
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

wandb.finish()