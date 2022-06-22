import argparse
import pathlib
import time
import os
import random

import torch
import gc
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import ConfusionMatrix

# from datasets.solidletters import SolidLetters
from datasets.assemblybodies import AssemblyBodies
from uvnet.models import Classification

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser("UV-Net solid model classification")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument("--dataset", choices=("assembly_bodies",), help="Dataset to train on")
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

if args.traintest == "train":
    # Train/val
    print(
        f"""
-----------------------------------------------------------------------------------
UV-Net Classification
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )

    continued_training = False

    if args.checkpoint:
        print("Loading from previous checkpoint - continuing previous training")
        model = Classification.load_from_checkpoint(args.checkpoint)
        continued_training = True
    else:
        model = Classification(num_classes=Dataset.num_classes())

    ##########################################################################
    """Generating our own train.txt and test.txt (randomly)"""

    if not continued_training:
        train_percentage = 0.8

        all_bins = os.listdir(args.dataset_path)
        for i in range(len(all_bins)):
            all_bins[i] = all_bins[i].split(".bin")[0]

        random.shuffle(all_bins)
        train_bins = all_bins[:int(len(all_bins) * train_percentage)]
        test_bins = all_bins[int(len(all_bins) * train_percentage):]

        with open(args.dataset_path + "/train.txt", "w") as f:
            for line in train_bins:
                f.write(str(line) + '\n')

        with open(args.dataset_path + "/test.txt", "w") as f:
            for line in test_bins:
                f.write(str(line) + '\n')

        print("Total number of samples:", len(all_bins))
        print("Total number of training samples:", len(train_bins))
        print("Total number of testing samples:", len(test_bins))
        print("-----------------------------------------------------------------------------------")

    else:
        print("Utilizing the train-test split from previous experiment")

    ##########################################################################

    train_data = Dataset(root_dir=args.dataset_path, split="train")
    val_data = Dataset(root_dir=args.dataset_path, split="val")
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

else:
    # Test
    assert (
            args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"

    test_data = Dataset(root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = Classification.load_from_checkpoint(args.checkpoint)
    test_acc = trainer.test(model=model, test_dataloaders=test_loader, verbose=True)
    # test_results = trainer.predict(model=model, dataloaders=test_loader)

    # for batch in test_loader:
    #     print(batch)

    # print("Classification results on test set:", results)
