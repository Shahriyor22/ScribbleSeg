import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UNetResNet34(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = smp.Unet(encoder_name = "resnet34", encoder_weights = "imagenet", in_channels = 5, classes = 1)

    def forward(self, x, target_size = None):
        logits = self.net(x)

        if target_size is None:
            return logits
        else:
            return F.interpolate(logits, size = target_size, mode = "bilinear", align_corners = False)
