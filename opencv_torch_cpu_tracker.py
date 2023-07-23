import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU

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


# load your model here

model = torch.load('./model/model.pth', map_location=torch.device('cpu'))
model.eval()

# open webcam feed
cap = cv2.VideoCapture('./cat.mp4')

try:
    while True:
        # read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess frame for your model
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))  # resize to your model's expected input size
        image = image.transpose((2, 0, 1))  # convert HWC -> CHW
        image = image / 255.  # normalize pixel values if necessary
        image = torch.Tensor(image)  # convert to tensor
        image = image.unsqueeze(0)  # add batch dimension

        # apply model to frame
        with torch.no_grad():
            logits = model(image)
            pr_mask = logits.sigmoid().squeeze().numpy()

        # postprocess mask for display
        pr_mask = cv2.resize(pr_mask, (frame.shape[1], frame.shape[0]))  # resize to original frame size
        pr_mask = np.uint8(pr_mask * 255)  # scale to [0, 255]
        pr_mask_color = cv2.applyColorMap(pr_mask, cv2.COLORMAP_JET)  # colorize mask for better visibility

        # display mask
        cv2.imshow('Segmentation Mask', pr_mask_color)

        # break on ESC key
        if cv2.waitKey(1) == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()