#!/usr/bin/python
"""make custom Dataset Class
"""

from typing import Any, Optional

from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import librosa


class AudioDataset(Dataset):
    def __init__(
        self,
        labels: np.ndarray,
        audio_paths: np.ndarray,
        sampling_rate: int,
        spec_mode: str = "melspectrogram",
        ref=np.min,
        transforms: Optional[v2.Compose] = None,
    ) -> None:
        """initialize of custom datasets.

        Args:
            labels (np.ndarray): datasets' label.
            audio_paths (np.ndarray): array which include path to audio data.
            sampling_rate (int): sampling rate to load audio data.
            transforms (v2.Compose, optional): transform for audio-image tensor. Defaults to v2.Compose( [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)] ).
        """
        self.labels = labels
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate
        self.spec_mode = spec_mode
        self.ref = ref
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[index]
        audio_path = self.audio_paths[index]
        audio, _ = librosa.load(audio_path)

        # TODO:Add other method to convert image.
        if self.spec_mode == "melspectrogram":
            spec = self.melspectrogram_transform(audio)

        if self.transforms is not None:
            spec = self.transforms(spec)

        return (spec, label)

    def melspectrogram_transform(self, audio: np.ndarray) -> np.ndarray:
        """transform audio array to melspectrogram image tensor.

        Args:
            audio (np.ndarray): audio array.

        Returns:
            np.ndarray: melspectrogram image tensor.
        """
        spec = librosa.feature.melspectrogram(y=audio, sr=self.sampling_rate)
        spec = librosa.power_to_db(spec, ref=self.ref)
        spec = np.stack([spec] * 3)
        spec = torch.tensor(spec)
        return spec


if __name__ == "__main__":
    TRAIN_AUDIO_PATH = "/workdir/mount/data/train_audio"
    import pandas as pd
    from torch.utils.data import DataLoader

    metadeta_df = pd.read_csv("/workdir/mount/data/train_metadata.csv")
    metadeta_df["filename"] = metadeta_df["filename"].apply(
        lambda x: f"{TRAIN_AUDIO_PATH}/{x}"
    )

    dataset = AudioDataset(
        labels=metadeta_df["primary_label"].to_numpy(),
        audio_paths=metadeta_df["filename"].to_numpy(),
        sampling_rate=22050,
        transforms=v2.Compose(
            [
                v2.RandomCrop((128, 128)),
                v2.Resize((224, 224)),
                v2.Lambda(lambda x: x / x.max()),
            ]
        ),
    )

    img = dataset[100][0].permute(1, 2, 0)
    print(img.size())
