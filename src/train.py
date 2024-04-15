#!/root/.pyenv/versions/3.11.5/bin/python
import argparse
from argparse import ArgumentParser
from collections import OrderedDict
from curses import meta
import os
from typing import Type

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import AverageMeter, accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import wandb

from models import Toymodel
from utils.dataset import AudioDataset

parser = ArgumentParser()
parser.add_argument(
    "--epochs",
    metavar="EPOCHS",
    type=int,
    default=300,
)
parser.add_argument(
    "--lr",
    metavar="LEANING_RATE",
    type=int,
    default=0.0001,
    help="""help me !!!!""",
)

parser.add_argument(
    "--data_dir",
    metavar="DATA-DIR",
    type=str,
    help="""""",
)

args = parser.parse_args()


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize of W&B

    # load train_metadata.csv
    meta_df = pd.read_csv(os.path.join(args.data_dir, "train_metadata"))

    # apply audio data path to DataFrame
    meta_df["filename"] = meta_df["filename"].apply(
        lambda x: f"{args.data_dir}/train_audio/{x}"
    )

    # apply LabelEncoding to label.
    encoder = LabelEncoder()
    meta_df["primary_label"] = encoder.fit_transform(meta_df["primary_label"])

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

    # make model instance
    model = Toymodel()
    model = model.to(DEVICE)

    # optimizer setup
    optimizer = create_optimizer_v2(
        model.parameters,
        opt="adamw",
        lr=args.lr,
        weight_decay=0,
        momentum=0.9,
    )

    # scheduler setup
    scheduler = create_scheduler_v2(optimizer, "cosine")

    # loss function setup
    # TODO handle with mixuped inputs
    train_loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.CrossEntropyLoss()

    # train&eval
    for epoch in tqdm(range()):

        train_one_epoch(
            epoch,
            model,
            train_loader,
            train_loss_fn,
            optimizer,
        )


def train_one_epoch(
    epoch: int,
    model: Type[nn.Module],
    loader,
    loss_fn,
    optimizer,
    device: torch.device,
) -> None:
    model.train()
    optimizer.zero_grad()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


def eval(
    epoch: int,
    model: Type[nn.Module],
    loader: Type[DataLoader],
    loss_fn,
    device: torch.device,
) -> OrderedDict:
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    top10_m = AverageMeter()

    model.eval()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        top1, top5, top10 = accuracy(outputs, labels, topk=(1, 5, 10))

        loss_m.update(loss.item(), inputs.size(0))
        top1_m.update(top1.item(), outputs.size(0))
        top5_m.update(top5.item(), outputs.size(0))
        top10_m.update(top10.item(), outputs.size(0))

    metrics = OrderedDict(
        [
            ("loss", loss_m),
            ("top1", top1_m),
            ("top5", top5_m),
            ("top10", top10_m),
        ]
    )

    return metrics


if __name__ == "__main__":

    main()
