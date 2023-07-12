import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
root = "."

train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")
test_dataset = SimpleOxfordPetDataset(root, "test")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=1, shuffle=False, num_workers=2)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=2)


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


arch = "FPN"
encoder_name = "resnet34"
in_channels = 3
out_classes = 1
model = PetModel(arch, encoder_name, in_channels, out_classes)
criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

def give_model():
    return model

def give_data_loaders():
    return train_dataloader, valid_dataloader, test_dataloader

def give_criterion():
    return criterion