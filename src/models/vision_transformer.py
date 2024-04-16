#!/usr/bin/python
"""fine-tuned model class."""

import torch
import torch.nn as nn
from timm.models import create_model


class FineTunedVidionTransformer(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.backbone = create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        )

    def forward(self, x) -> torch.tensor:
        x = self.backbone(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = FineTunedVidionTransformer(num_classes=182)
    for param in model.parameters():
        param.requires_grad = False

    summary(model, (32, 3, 224, 224))
