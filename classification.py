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

from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_lightning.utilities.seed import seed_everything


# from datasets.solidletters import SolidLetters
from datasets.assemblybodies import AssemblyBodies, DROP_BODIES
from uvnet.models import Classification

random.seed(7)

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser("UV-Net solid model classification")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument("--dataset", choices=("assembly_bodies",), help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--fixed_split", type=bool, default=False, help="Fixed train-test split")
parser.add_argument("--use_existing_split", type=bool, default=False, help="Use existing train-test split files")
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
    seed_everything(workers=True)
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
        print("Loading from existing checkpoint - continuing previous training")
        model = Classification.load_from_checkpoint(args.checkpoint)
        continued_training = True
    else:
        model = Classification(num_classes=Dataset.num_classes())

    ##########################################################################
    """Generating our own train.txt and test.txt (randomly)"""

    if not continued_training and not args.use_existing_split:
        train_bins, test_bins = [], []

        if args.fixed_split:
            print("Utilizing pre-defined fixed train test split")
            # use pre-defined and fixed train-test split from dataset
            with open(args.dataset_path + '\\' + "assemblies_train_val.txt") as f:
                lines = f.readlines()
                assemblies_train_val = [line.strip() for line in lines]

            with open(args.dataset_path + '\\' + "assemblies_test.txt") as f:
                lines = f.readlines()
                assemblies_test = [line.strip() for line in lines]

            with open(args.dataset_path + '\\' + "bodies_train_val.txt") as f:
                lines = f.readlines()
                bodies_train_val = [line.strip() for line in lines]

            with open(args.dataset_path + '\\' + "bodies_test.txt") as f:
                lines = f.readlines()
                bodies_test = [line.strip() for line in lines]

            all_bins = os.listdir(args.dataset_path)
            for i in range(len(all_bins)):
                all_bins[i] = all_bins[i].split(".bin")[0]

            for bin in all_bins:
                if ".txt" in bin:
                    continue
                assembly_id = bin.split("_sep_")[1]
                body_id = bin.split("_sep_")[2]

                if assembly_id in assemblies_train_val and body_id in bodies_train_val:
                    train_bins.append(bin)
                elif assembly_id in assemblies_test and body_id in bodies_test:
                    test_bins.append(bin)
                else:
                    print(bin)
                    print("Unexpected split: no graph in train should have any of its bodies in test, vice versa")
                    exit(1)

            random.shuffle(train_bins)
            random.shuffle(test_bins)

        else:
            # randomly perform train-test split (8:2)
            print("Shuffling the dataset to create randomized train test split")
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

    if continued_training or args.use_existing_split:
        print("Utilizing existing train-test split (no changing of train.txt and test.txt)")

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

    """Testing"""
    # trainer.test(model=model, test_dataloaders=[test_loader], verbose=True)

    """Predictions - Classification Report & Confusion Matrix"""

    predictions, ground_truths = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference on test loader"):
            preds, labels = model.test_step(batch, None)
            preds = preds.tolist()
            labels = labels.tolist()
            predictions.append(preds)
            ground_truths.append(labels)

    predictions = list(numpy.concatenate(predictions).flat)
    ground_truths = list(numpy.concatenate(ground_truths).flat)

    print(classification_report(y_pred=predictions, y_true=ground_truths))
    cf = confusion_matrix(y_pred=predictions, y_true=ground_truths, normalize="true")
    plt.figure(figsize=(24, 18))

    if not DROP_BODIES:
        label = ["Metal_Aluminum",
                 "Metal_Ferrous",
                 "Metal_Ferrous_Steel",
                 "Metal_Non-Ferrous",
                 "Other",
                 "Paint",
                 "Plastic",
                 "Wood"]
    else:
        label = ["Metal_Aluminum",
                 "Metal_Ferrous",
                 "Metal_Non-Ferrous",
                 "Other",
                 "Plastic",
                 "Wood"]

    sn.heatmap(cf, annot=True, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label, annot_kws={"size": 25})
    plt.xticks(size='xx-large', rotation=45)
    plt.yticks(size='xx-large', rotation=45)
    plt.tight_layout()

    print("Confusion Acc = ", round(sum(cf.diagonal() / cf.sum(axis=1)) / len(cf), 3))

    plt.savefig(fname=f'confusion_matrix.png', format='png')
    plt.savefig(fname=f'confusion_matrix.pdf', format='pdf')
