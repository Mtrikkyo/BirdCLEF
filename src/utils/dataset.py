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
        spec_mode="mel-spectrogram",
        ref=np.min,
        transforms: v2.Compose = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),
    ) -> None:
        """initialize of custom datasets.

        Args:
            labels (np.ndarray): _description_
            audio_paths (np.ndarray): _description_
            sampling_rate (int): _description_
            transforms (v2.Compose, optional): _description_. Defaults to v2.Compose( [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)] ).
        """
        self.labels = labels
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate
        self.ref = ref
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[index]
        audio_path = self.audio_paths[index]
        audio, _ = librosa.load(audio_path)

        if self.spec_mode == "mel-spectrogram":
            spec = self._convert_melspectrogram(audio)

        image = self.transforms(spec)
        return (image, label)

    def _convert_melspectrogram(self, audio: np.ndarray) -> np.ndarray:

        spec = librosa.feature.melspectrogram(audio, self.sampling_rate)
        spec = librosa.power_to_db(spec, ref=self.ref)

        return spec


if __name__ == "__main__":
    import pandas as pd

    metadeta_df = pd.read_csv("/workdir/mount/data/train_metadata.csv")
