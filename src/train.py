#!/root/.pyenv/versions/3.11.5/bin/python
import argparse
from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader
from timm.scheduler import create_scheduler_v2
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

from models import Toymodel
from utils.dataset import AudioDataset

parser = ArgumentParser()
parser.add_argument(
    "--data_dir",
    metavar="DATA-DIR",
    type=str,
    help="""""",
)

args = parser.parse_args()


def main():
    # load train_metadata.csv
    meta_df = pd.read_csv(os.path.join(args.data_dir, "train_metadata"))

    # apply audio data path to DataFrame
    meta_df["filename"] = meta_df["filename"].apply(
        lambda x: f"{args.data_dir}/train_audio/{x}"
    )

    # make dataset
    dataset = AudioDataset(
        labels=meta_df["primary_label"].to_numpy(),
        audio_paths=meta_df["filename"].to_numpy(),
        sampling_rate=20050,
    )

    # train_test_split
    train_dataset, valid_dataset = train_test_split(dataset, test_size=1 / 6)

    # make dataloader
    train_loader = DataLoader(train_dataset)
    valid_loader = DataLoader(valid_dataset)
