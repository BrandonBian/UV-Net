import pathlib
import string

import torch
import dgl
from sklearn.model_selection import train_test_split
from collections import Counter

from datasets.base import BaseDataset

DROP_BODIES = True

if not DROP_BODIES:
    LABEL_MAPPING = {
        "Metal_Aluminum": 0,
        "Metal_Ferrous": 1,
        "Metal_Ferrous_Steel": 2,
        "Metal_Non-Ferrous": 3,
        "Other": 4,
        "Paint": 5,
        "Plastic": 6,
        "Wood": 7
    }
else:
    LABEL_MAPPING = {
        "Metal_Aluminum": 0,
        "Metal_Ferrous": 1,
        "Metal_Non-Ferrous": 2,
        "Other": 3,
        "Plastic": 4,
        "Wood": 5
    }


def get_distribution(files, type):
    """Get the distribution of labels"""
    labels = []
    for file in files:
        labels.append(str(file).split('\\')[-1].split("_sep_")[0])

    cnt = Counter(labels)
    distribution = [(i, round((cnt[i] / len(labels) * 100.0), 3)) for i, count in cnt.most_common()]
    print(f"Distribution for [{type}] set: {distribution}")


def _get_filenames(root_dir, filelist):
    with open(str(root_dir / f"{filelist}"), "r") as f:
        file_list_ = [x.strip() for x in f.readlines()]

    if DROP_BODIES:
        file_list = []
        for file in file_list_:
            label = str(file).split('\\')[-1].split("_sep_")[0]
            if label == "Paint" or label == "Metal_Ferrous_Steel":
                continue
            file_list.append(file)
    else:
        file_list = file_list_

    files = list(
        x
        for x in root_dir.rglob(f"*.bin")
        if x.stem in file_list
        # if util.valid_font(x) and x.stem in file_list
    )
    return files


def to_label(string):
    label = str(string).split('\\')[-1].split("_sep_")[0]
    return LABEL_MAPPING[label]


class AssemblyBodies(BaseDataset):
    @staticmethod
    def num_classes():
        return len(LABEL_MAPPING)

    def __init__(
            self,
            root_dir,
            split="train",
            center_and_scale=True,
            random_rotate=False,
    ):
        """
        Load the Graph BIN dataset

        Args:
            root_dir (str): Root path to the dataset
            split (str, optional): Split (train, val, or test) to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        path = pathlib.Path(root_dir)

        self.random_rotate = random_rotate
        self.label_set = None

        if split in ("train", "val"):
            file_paths = _get_filenames(path, filelist="train.txt")

            labels = [to_label(fn) for fn in file_paths]
            train_files, val_files = train_test_split(
                file_paths, test_size=0.2, random_state=42, stratify=labels,
            )

            if split == "train":
                file_paths = train_files
                get_distribution(file_paths, "train")

            elif split == "val":
                file_paths = val_files
                get_distribution(file_paths, "val")

        elif split == "test":
            file_paths = _get_filenames(path, filelist="test.txt")
            get_distribution(file_paths, "test")

        elif split == "train_val":
            file_paths = _get_filenames(path, filelist="train.txt")
            get_distribution(file_paths, "train_val")

        else:
            print("ERROR: incorrect split!")
            exit(1)

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally get the label from the filename and store it in the sample dict
        sample["label"] = torch.tensor([to_label(file_path)]).long()
        return sample

    def _collate(self, batch):
        collated = super()._collate(batch)
        collated["label"] = torch.cat([x["label"] for x in batch], dim=0)
        return collated
