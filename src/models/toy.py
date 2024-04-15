import torch
import torch.nn as nn


class Toymodel(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1, -1),
            nn.Linear(64, 182),
        )


if __name__ == "__main__":
    from torchinfo import summary

    model = Toymodel()
    summary(model, input_size=(3, 224, 224), batch_dim=0)
