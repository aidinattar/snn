"""
generate_norb_dataset.py

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

import os
import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import imwrite

parser = argparse.ArgumentParser(description='Generate NORB dataset.')
parser.add_argument('--train_size', type=int, default=24300, help='Size of the training set')
parser.add_argument('--test_size', type=int, default=24300, help='Size of the test set')
parser.add_argument('--dataset_root', type=str, default='norb_mat', help='Path to the NORB dataset root directory')
parser.add_argument('--export_dir', type=str, default='norb', help='Directory to export the generated dataset')
args = parser.parse_args()

class SmallNORBDataset:
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


if __name__ == '__main__':
    # read the dataset from the raw files
    # divide the dataset into training and test sets and save them in separate directories
    # norb/train and norb/test
    # each image is saved as a jpg file, in a subdirectory named after the category
    # the name of the image is the index of the image in the dataset
    train_size = args.train_size
    test_size = args.test_size
    dataset_root = args.dataset_root
    export_dir = args.export_dir

    dataset = SmallNORBDataset(dataset_root)
    dataset.export_to_jpg(export_dir, train_size, test_size)

# End of file generate_norb_dataset.py