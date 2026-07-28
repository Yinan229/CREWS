"""WiFi sensing dataset loading and domain splitting."""
from __future__ import annotations

import os

import numpy as np
import scipy.io as scio
import torch
import torch.utils.data as Data
from torch.utils.data import ConcatDataset, random_split


def generate_train_paths_New(image_base_dir, max_last_id=5):
    img_paths = []
    img_labels1 = []

    for fname in sorted(os.listdir(image_base_dir)):
        if fname == "figure":
            continue
        if not fname.endswith(".mat"):
            continue

        # e.g. "userch-1-1-1-10.mat"
        stem = os.path.splitext(fname)[0]
        parts = stem.split("-")

        try:
            last_id = int(parts[-1])
        except (ValueError, IndexError):
            continue

        if last_id > max_last_id:
            continue

        try:
            label = int(parts[1]) - 1
        except (ValueError, IndexError):
            continue

        img_paths.append(os.path.join(image_base_dir, fname))
        img_labels1.append(label)

    return img_paths, img_labels1


def WiFi_Folder_New(path):
    train_video_feature = []
    train_video_label1 = []
    i = 1
    img_paths, img_labels1 = generate_train_paths_New(path)
    for dataFile in img_paths:
        print(dataFile)

        data_1 = scio.loadmat(dataFile)
        data_array_1 = np.zeros((8, 100, 800), dtype=np.float32)
        data_array_1[:, :, :data_1['dfs_sp'].shape[2]] = data_1['dfs_sp']
        train_video_feature.append(data_array_1)
        train_video_label1.append(img_labels1[i - 1])

        i = i + 1

    train_video_feature = np.array(train_video_feature)
    train_video_label1 = np.array(train_video_label1)
    train_video_feature = torch.from_numpy(train_video_feature)
    train_video_label1 = torch.from_numpy(train_video_label1)
    train_dataset = Data.TensorDataset(train_video_feature, train_video_label1)

    return train_dataset


class MultipleDomainDataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 0
    ENVIRONMENTS = None
    INPUT_SHAPE = None

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentImageFolder_WiFi_New(MultipleDomainDataset):
    def __init__(self, root, environments=None, num_classes=6):
        super().__init__()
        if environments is None:
            environments = [
                'data_for_yn_V1', 'data_for_ch_V1', 'data_for_zj_V1',
                'data_for_wt_V1', 'data_for_wq_V1', 'data_for_yl_V1',
            ]
        self.datasets = []

        for i, environment in enumerate(environments):
            path = os.path.join(root, environment)
            env_dataset = WiFi_Folder_New(path)
            self.datasets.append(env_dataset)
            print('finish %d' % i)

        self.input_shape = (1, 100, 800,)
        self.num_classes = num_classes


def split_each_domain(dataset_by_env, test_ratio=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)

    train_parts, test_parts = [], []
    for env_id in range(len(dataset_by_env)):
        ds = dataset_by_env[env_id]
        n = len(ds)
        n_test = int(round(test_ratio * n))
        n_train = n - n_test
        ds_train, ds_test = random_split(ds, [n_train, n_test], generator=g)
        train_parts.append(ds_train)
        test_parts.append(ds_test)

    train_ds = ConcatDataset(train_parts)
    test_ds = ConcatDataset(test_parts)
    return train_ds, test_ds


def build_train_test_datasets(root, environments=None, num_classes=6,
                              test_ratio=0.3, split_seed=2025):
    """Load all environments and produce concatenated train/test datasets."""
    full = MultipleEnvironmentImageFolder_WiFi_New(
        root, environments=environments, num_classes=num_classes
    )
    train_dataset, test_dataset = split_each_domain(
        full, test_ratio=test_ratio, seed=split_seed
    )
    return train_dataset, test_dataset
