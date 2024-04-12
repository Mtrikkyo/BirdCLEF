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
        labels,
        audio_paths,
        sampling_rate: int,
        transforms: v2.Compose = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),
    ) -> None:

        self.labels = labels
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Any:
        label = self.labels[index]
        audio_path = self.audio_paths[index]
        audio, _ = librosa.load(audio_path)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sampling_rate)
        mel_spectrogram_dB = librosa.power_to_db(mel_spectrogram, ref=np.min)

        image = self.transforms(mel_spectrogram_dB)
        return image, label
