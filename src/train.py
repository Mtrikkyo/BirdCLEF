#!/root/.pyenv/versions/3.11.5/bin/python
from argparse import ArgumentParser
from collections import OrderedDict
import os
from typing import Type

import pandas as pd
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import AverageMeter, accuracy, update_summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import wandb

from models import Toymodel
from utils.dataset import AudioDataset

parser = ArgumentParser()
parser.add_argument(
    "-e",
    "--epochs",
    metavar="EPOCHS",
    type=int,
    default=300,
)
parser.add_argument(
    "--lr",
    metavar="LEANING_RATE",
    type=float,
    default=0.0001,
    help="""help me !!!!""",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    help="""number of data instaces per mini-batch.""",
)

parser.add_argument(
    "-d",
    "--data_dir",
    metavar="DATA-DIR",
    type=str,
    help="""Path to the directory where the data is stored.""",
)

parser.add_argument(
    "-s",
    "--save_dir",
    type=str,
    help="""Path to the directory where the data is stored.""",
)

args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(DEVICE)

    # load train_metadata.csv
    meta_df = pd.read_csv(os.path.join(args.data_dir, "train_metadata.csv"))

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
    print("making dataset is over.")

    # train_test_split
    train_dataset, valid_dataset = train_test_split(dataset, test_size=1 / 6)
    train_dataset = train_dataset.to(DEVICE)
    valid_dataset = valid_dataset.to(DEVICE)

    # make dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
    )
    print("making train_loader is over.")
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
    )

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
    scheduler = create_scheduler_v2(
        optimizer,
        "cosine",
        warmup_prefix=True,
    )

    # loss function setup
    # TODO handle with mixuped inputs
    train_loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.CrossEntropyLoss()

    # initialize of W&B
    run = wandb.init(project="BirdCLEF")
    run.save()
    artifact = wandb.Artifact("model", type="model")

    # train&eval
    for epoch in tqdm(range()):
        scheduler.step(epoch)

        train_one_epoch(
            epoch,
            model,
            train_loader,
            train_loss_fn,
            optimizer,
        )

        train_metrics = eval(
            epoch,
            model,
            train_loader,
            train_loss_fn,
        )

        eval_metrics = eval(
            epoch,
            model,
            valid_loader,
            valid_loss_fn,
        )

        lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        update_summary(
            epoch=epoch,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            filename=os.path.join(args.save_dir, "summary.csv"),
            lr=sum(lrs) / len(lrs),
            write_header=epoch == 0,
            log_wandb=True,
        )
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))
        artifact.add_file(os.path.join(args.save_dir, "model.pt"))
        run.log_artifact(artifact)
    run.finish()


def train_one_epoch(
    epoch: int,
    model: Type[nn.Module],
    loader,
    loss_fn,
    optimizer,
) -> None:
    model.train()
    optimizer.zero_grad()

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


def eval(
    epoch: int,
    model: Type[nn.Module],
    loader: Type[DataLoader],
    loss_fn,
) -> OrderedDict:
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    top10_m = AverageMeter()

    model.eval()

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

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
