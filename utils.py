################################################################################
# Title:            utils.py                                                   #
# Description:      Some utility functions                                     #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.2                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import os
import torch
import struct
import datetime
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from s1c1 import S1C1
from SpykeTorch import utils
from tqdm import tqdm
from imageio import imwrite

def get_time():
    """Get the current time"""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_time_stamp():
    """Get the current time stamp"""
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def get_time_stamp_ms():
    """Get the current time stamp with milliseconds"""
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

def find_percentile(arr, value):
    """
    Find the percentile of a value in an array

    Parameters
    ----------
    arr : list
        The array of values
    value : float
        The value to find the percentile of

    Returns
    -------
    float
        The percentile of the value in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    rank = np.sum(sorted_arr <= value)
    percentile = (rank / len(arr)) * 100
    
    return percentile

def find_percentile_index(arr, percentile):
    """
    Find the index of a percentile in an array

    Parameters
    ----------
    arr : list
        The array of values
    percentile : float
        The percentile to find the index of

    Returns
    -------
    int
        The index of the percentile in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    index = int((percentile / 100) * len(arr))
    
    return index

def find_percentile_value(arr, percentile):
    """
    Find the value of a percentile in an array

    Parameters
    ----------
    arr : list
        The array of values
    percentile : float
        The percentile to find the value of

    Returns
    -------
    float
        The value of the percentile in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    index = int((percentile / 100) * len(arr))
    value = sorted_arr[index]
    
    return value

def find_percentile_range(arr, percentile):
    """
    Find the range of a percentile in an array

    Parameters
    ----------
    arr : list
        The array of values
    percentile : float
        The percentile to find the range of

    Returns
    -------
    tuple
        The range of the percentile in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    index = int((percentile / 100) * len(arr))
    value = sorted_arr[index]
    lower = sorted_arr[:index]
    upper = sorted_arr[index:]
    lower_range = (np.min(lower), np.max(lower))
    upper_range = (np.min(upper), np.max(upper))
    
    return lower_range, upper_range

def get_embeddings_metadata(model, dataloader, device):
    """
    Get embeddings and metadata from a model and dataloader

    Parameters
    ----------
    model : torch.nn.Module
        The model to get embeddings from
    dataloader : torch.utils.data.DataLoader
        The dataloader to get metadata from
    device : torch.device
        The device to use

    Returns
    -------
    torch.Tensor
        The embeddings
    list
        The metadata
    """
    model.eval()
    embeddings = []
    metadata = []
    label_imgs = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            embeddings.append(output)
            metadata.extend(target.cpu().numpy())
            label_imgs.append(data.cpu())
    
    embeddings = torch.cat(embeddings)
    label_imgs = torch.cat(label_imgs)
    return embeddings, metadata, label_imgs

def prepare_data(dataset, batch_size):
    """
    Prepare the data for training
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to use
    batch_size : int
        Batch size for the data loader

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Data loader for the training set
    test_loader : torch.utils.data.DataLoader
        Data loader for the test set
    num_classes : int
        Number of classes in the dataset
    """
    kernels = [
        utils.DoGKernel(3,3/9,6/9),
        utils.DoGKernel(3,6/9,3/9),
        utils.DoGKernel(7,7/9,14/9),
        utils.DoGKernel(7,14/9,7/9),
        utils.DoGKernel(13,13/9,26/9),
        utils.DoGKernel(13,26/9,13/9)
    ]

    filter = utils.Filter(kernels, padding=6, thresholds=50)
    s1c1 = S1C1(filter, timesteps=15)

    # Load dataset
    if dataset == "mnist":
        data_root = "data_mnist"
        num_classes = 10
        train_data = utils.CacheDataset(
            torchvision.datasets.MNIST(
                root = data_root,
                train = True,
                download = True,
                transform = s1c1
            )
        )
        test_data = utils.CacheDataset(
            torchvision.datasets.MNIST(
                root = data_root,
                train = False,
                download = True,
                transform = s1c1
            )
        )
    elif dataset == "cifar10":
        num_classes = 10
        data_root = "data_cifar10"
        train_data = utils.CacheDataset(
            torchvision.datasets.CIFAR10(
                root = data_root,
                train = True,
                download = True,
                transform = s1c1
            )
        )
        test_data = utils.CacheDataset(
            torchvision.datasets.CIFAR10(
                root = data_root,
                train = False,
                download = True,
                transform = s1c1
            )
        )
    elif dataset == "emnist":
        num_classes = 47
        data_root = "data_emnist"
        train_data = utils.CacheDataset(
            torchvision.datasets.EMNIST(
                root = data_root,
                split = "digits",
                train = True,
                download = True,
                transform = s1c1
            )
        )
        test_data = utils.CacheDataset(
            torchvision.datasets.EMNIST(
                root = data_root,
                split = "digits",
                train = False,
                download = True,
                transform = s1c1
            )
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = len(test_data),
        shuffle = False
    )
    metrics_data = torch.utils.data.Subset(
        test_data,
        range(batch_size)
    )
    metrics_loader = torch.utils.data.DataLoader(
        metrics_data,
        batch_size = len(metrics_data),
        shuffle = False
    )
    return train_loader, test_loader, metrics_loader, num_classes

def is_model_on_cuda(model):
    """Check if the model is on CUDA"""
    return next(model.parameters()).is_cuda


class SmallNORBDataset:
    """
    Code partially taken from https://github.com/ndrplz/small_norb.git

    This script generates the NORB dataset from the raw data files. The NORB dataset
    is a dataset of stereo images of 3D objects. The dataset is available at
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/. The dataset is divided into
    two parts: a small dataset and a large dataset. The small dataset contains 24300
    training examples and 24300 test examples. The large dataset contains 24300
    training examples and 24300 test examples. The small dataset is used in this
    script.

    The dataset is stored for each example as a 96x96 image. The images are stored
    in jpegs, so they need to be decoded.

    The dataset is stored in a binary format. The training set is stored in a file
    called 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat'. The test set is
    stored in a file called 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat'.
    The labels for the training set are stored in a file called
    'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'. The labels for the test set
    are stored in a file called 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'.
    """
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.dataset_files = self._get_dataset_files()
        self.data = self._load_data()

    def _get_dataset_files(self):
        files = ['cat', 'info', 'dat']
        prefixes = {
            'train': 'smallnorb-5x46789x9x18x6x2x96x96-training',
            'test': 'smallnorb-5x01235x9x18x6x2x96x96-testing'
        }
        dataset_files = {split: {f: os.path.join(self.dataset_root, f'{prefixes[split]}-{f}.mat') for f in files} for split in ['train', 'test']}
        return dataset_files

    def _load_data(self):
        data = {split: [self._load_example(i, split) for i in tqdm(range(24300), desc=f'Loading {split} data')] for split in ['train', 'test']}
        return data

    def _load_example(self, i, split):
        example = {}
        example['category'] = self._load_category(i, split)
        example['info'] = self._load_info(i, split)
        example['images'] = self._load_images(i, split)
        return example

    def _load_category(self, i, split):
        with open(self.dataset_files[split]['cat'], 'rb') as f:
            f.seek(i*4+20)
            category, = struct.unpack('<i', f.read(4))
        return category

    def _load_info(self, i, split):
        with open(self.dataset_files[split]['info'], 'rb') as f:
            f.seek(i*16+20)
            info = struct.unpack('<4i', f.read(16))
        return info

    def _load_images(self, i, split):
        with open(self.dataset_files[split]['dat'], 'rb') as f:
            f.seek(i*2*96*96+24)
            images = np.fromfile(f, dtype=np.uint8, count=2*96*96).reshape(2, 96, 96)
        return images

    def show_random_examples(self, split):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        for example in np.random.choice(self.data[split], 5):
            fig.suptitle(f'Category: {example["category"]} Info: {example["info"]}')
            axes[0].imshow(example['images'][0], cmap='gray')
            axes[1].imshow(example['images'][1], cmap='gray')
            plt.waitforbuttonpress()
            plt.cla()

    def export_to_jpg(self, export_dir, train_size, test_size):
        for split in ['train', 'test']:
            split_dir = os.path.join(export_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Delete everything in the split directory
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    sub_dir = os.path.join(root, dir)
                    for sub_root, sub_dirs, sub_files in os.walk(sub_dir):
                        for sub_file in sub_files:
                            os.remove(os.path.join(sub_root, sub_file))
                        for sub_sub_dir in sub_dirs:
                            os.rmdir(os.path.join(sub_root, sub_sub_dir))
                    os.rmdir(sub_dir)

            if split == 'train':
                size = train_size
            else:
                size = test_size
            for i, example in enumerate(tqdm(self.data[split][:size], desc=f'Exporting {split} images to {export_dir}')):
                for j, image in enumerate(example['images']):
                    if not os.path.exists(os.path.join(split_dir, str(example['category']))):
                        os.makedirs(os.path.join(split_dir, str(example['category'])), exist_ok=True)
                    # imwrite(os.path.join(split_dir, f'{i:06d}_{example["category"]}_{example["info"][0]}_{j}.jpg'), image)
                    imwrite(os.path.join(split_dir, str(example['category']), f'{i:06d}_{example["category"]}_{example["info"][0]}_{j}.jpg'), image)

def generate_norb_dataset(train_size, test_size, dataset_root, export_dir):
    dataset = SmallNORBDataset(dataset_root)
    dataset.export_to_jpg(export_dir, train_size, test_size)