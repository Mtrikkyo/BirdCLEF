#!/root/.pyenv/versions/3.11.5/bin/python
"""make custom Dataset Class
"""

from typing import Any

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
        transforms: v2.Compose = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),
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

        if self.spec_mode == "melspectrogram":
            spec = self.melspectrogram_transform(audio)

        image = self.transforms(spec)
        return (image, label)

    def melspectrogram_transform(self, audio: np.ndarray) -> np.ndarray:
        """transform audio array to melspectrogram image tensor.

        Args:
            audio (np.ndarray): audio array.

        Returns:
            np.ndarray: melspectrogram image tensor.
        """
        spec = librosa.feature.melspectrogram(audio, self.sampling_rate)
        spec = librosa.power_to_db(spec, ref=self.ref)

        return spec


if __name__ == "__main__":
    import pandas as pd

    metadeta_df = pd.read_csv("/workdir/mount/data/train_metadata.csv")
