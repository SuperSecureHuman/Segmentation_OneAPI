import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
import intel_extension_for_pytorch as ipex
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

root = "."

train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")
test_dataset = SimpleOxfordPetDataset(root, "test")

# It is a good practice to check datasets don`t intersects with each other
assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

class PetModel(nn.Module):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.std = torch.tensor(params["std"]).view(1, 3, 1, 1)
        self.mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    with tqdm(total=len(train_loader), desc="Training") as progress_bar:
        for batch in train_loader:
            #with torch.cpu.amp.autocast():
            images = batch["image"]  # Move images to CUDA device
            images = images
            masks = batch["mask"]
            masks = masks

            optimizer.zero_grad()
            logits_mask = model(images)
            loss = criterion(logits_mask, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.update(1)

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad(), tqdm(total=len(val_loader), desc="Validation") as progress_bar:
        for batch in val_loader:
            images = batch["image"]
            masks = batch["mask"]

            logits_mask = model(images)
            loss = criterion(logits_mask, masks)

            total_loss += loss.item()
            progress_bar.update(1)

    return total_loss / len(val_loader)


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad(), tqdm(total=len(test_loader), desc="Testing") as progress_bar:
        for batch in test_loader:
            images = batch["image"]
            masks = batch["mask"]

            logits_mask = model(images)
            loss = criterion(logits_mask, masks)

            total_loss += loss.item()
            progress_bar.update(1)

    return total_loss / len(test_loader)


arch = "FPN"
encoder_name = "resnet34"
in_channels = 3
out_classes = 1
model = PetModel(arch, encoder_name, in_channels, out_classes)
criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


model, optimizer = ipex.optimize(model=model, optimizer=optimizer, dtype=torch.float32)


num_epochs = 11
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss = validate(model, valid_dataloader, criterion)

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Testing
test_loss = test(model, test_dataloader, criterion)
print(f"Test Loss: {test_loss:.4f}")

# save model

torch.save(model.state_dict(), "model.pth")
