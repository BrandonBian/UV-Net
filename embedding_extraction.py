import argparse
import pathlib
import time
import os
import random

import torch
import gc
import numpy
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import shutil

from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import ConfusionMatrix
from sklearn.metrics import classification_report, confusion_matrix

# from datasets.solidletters import SolidLetters
from datasets.assemblybodies import AssemblyBodies
from uvnet.models import Classification

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser("UV-Net solid model classification")
parser.add_argument(
    "--traintest", choices=("train", "test"), default="test", help="Whether to train or test"
)
parser.add_argument("--dataset", choices=("assembly_bodies",), default="assembly_bodies", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="classification",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
    accumulate_grad_batches=32,
)

if args.dataset == "assembly_bodies":
    Dataset = AssemblyBodies
else:
    raise ValueError("Unsupported dataset")

if __name__ == "__main__":

    """Initialization"""
    args.checkpoint = r"results/test/0622/131533/last.ckpt"
    args.dataset_path = r"process/graphs"

    assert (
            args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"

    train_val_data = Dataset(root_dir=args.dataset_path, split="train_val")
    test_data = Dataset(root_dir=args.dataset_path, split="test")

    train_val_loader = train_val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = Classification.load_from_checkpoint(args.checkpoint)

    """Predictions - Generating Body Embeddings"""
    output_dir = "UV_embeddings.csv"
    if os.path.exists(output_dir):
        os.remove(output_dir)
    df = pd.DataFrame(list(), columns=["assembly_id", "body_id", "split", "embeddings"])
    df.to_csv(output_dir, index=False)

    train_val_cnt, test_cnt = 0, 0

    model.eval()
    with torch.no_grad():

        for batch in tqdm(train_val_loader, desc="Generating graph/body embeddings for [train + val loader]"):
            file_names, embeddings = model.test_embeddings(batch)
            for i in range(len(file_names)):
                train_val_cnt += 1

                assembly_id = file_names[i].split("_sep_")[1]
                body_id = file_names[i].split("_sep_")[-1]
                embedding = embeddings[i].tolist()

                csv_row = [assembly_id, body_id, "train-val", embedding]
                df = pd.DataFrame([csv_row], columns=["assembly_id", "body_id", "split", "embeddings"])
                df.to_csv(output_dir, mode='a', header=False, index=False)

        for batch in tqdm(test_loader, desc="Generating graph/body embeddings for [test loader]"):
            file_names, embeddings = model.test_embeddings(batch)
            for i in range(len(file_names)):
                test_cnt += 1

                assembly_id = file_names[i].split("_sep_")[1]
                body_id = file_names[i].split("_sep_")[-1]
                embedding = embeddings[i].tolist()

                csv_row = [assembly_id, body_id, "test", embedding]
                df = pd.DataFrame([csv_row], columns=["assembly_id", "body_id", "split", "embeddings"])
                df.to_csv(output_dir, mode='a', header=False, index=False)

    print(f"Generated embeddings for Train & Val = {train_val_cnt} | Test = {test_cnt}")