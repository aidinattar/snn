import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DopNetH5Dataset(Dataset):
    def __init__(self, h5_file, person_list, gestures, transform=None, target_width=144):
        self.h5_file = h5_file
        self.person_list = person_list
        self.gestures = gestures
        self.transform = transform
        self.target_width = target_width  # Fixed width to pad to
        self.data_index = self._build_index()

    def _build_index(self):
        """Builds an index for easy access to each sample."""
        index = []
        with h5py.File(self.h5_file, 'r') as h5f:
            for person in self.person_list:
                for gesture in self.gestures:
                    # Dynamically find all samples under each gesture
                    gesture_group = h5f[f"{person}/{gesture}"]
                    samples = list(gesture_group.keys())
                    for sample in samples:
                        index.append((person, gesture, sample))
        return index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # Open the HDF5 file in read mode
        with h5py.File(self.h5_file, 'r') as h5f:
            # Get the sample details
            person, gesture, sample = self.data_index[idx]
            data = h5f[f"{person}/{gesture}/{sample}"][:]

        # Pad or trim data to the target width
        padded_data = np.zeros((data.shape[0], self.target_width), dtype=np.float32)
        padded_data[:, :data.shape[1]] = data[:, :self.target_width]

        # Convert to tensor and apply transform if specified
        data = torch.tensor(data, dtype=torch.float32)
        label = gesture  # Use gesture as label
        if self.transform:
            data = self.transform(data)

        return data, label
    
class DopNetH5TestDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing test data.
            transform (callable, optional): Optional transform to apply.
        """
        self.h5_file = h5_file
        self.transform = transform
        self.data_index = self._build_index()

    def _build_index(self):
        """Builds an index for easy access to each test sample."""
        index = []
        with h5py.File(self.h5_file, 'r') as h5f:
            test_group = h5f
            samples = list(test_group.keys())
            for sample in samples:
                index.append(sample)
        return index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as h5f:
            sample = self.data_index[idx]
            data = h5f[f"{sample}"][:]

        # Convert to tensor and apply transform if specified
        data_tensor = torch.tensor(data, dtype=torch.float32)
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, -1  # Return -1 as a placeholder label if not available


def collate_fn(batch):
    # Find the maximum width in the batch
    # max_width = max(data.shape[1] for data, _ in batch)
    max_width = 540
    
    # Pad each sample in the batch to the max width
    padded_batch = []
    lengths = []
    for data, label in batch:
        lengths.append(data.shape[1])
        padded_data = np.zeros((data.shape[0], max_width), dtype=np.float32)
        padded_data[:, :data.shape[1]] = data
        padded_batch.append((torch.tensor(padded_data, dtype=torch.float32), label))
    
    # Separate data and labels for the batch
    data, labels = zip(*padded_batch)
    return torch.stack(data), torch.tensor(labels), lengths


def main():
    # Load the dataset and DataLoader
    h5_file = os.path.join('data', 'gesture', 'train', 'preprocessed_gesture_data_train.h5')
    person_list = ['A', 'B', 'C', 'D', 'E', 'F']
    gestures = [0, 1, 2, 3]  # Wave, Pinch, Swipe, Click
    dopnet_h5_dataset = DopNetH5Dataset(h5_file=h5_file, person_list=person_list, gestures=gestures)
    dopnet_h5_dataloader = DataLoader(dopnet_h5_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for batch_x, batch_y in dopnet_h5_dataloader:
        print("Batch spectrograms shape:", batch_x.shape)
        print("Batch labels:", batch_y)
        break

    # Load the test dataset and DataLoader
    test_h5_file = os.path.join('data', 'gesture', 'test', 'preprocessed_gesture_data_test.h5')
    dopnet_test_dataset = DopNetH5TestDataset(h5_file=test_h5_file)
    dopnet_test_dataloader = DataLoader(dopnet_test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    for test_batch_x, _ in dopnet_test_dataloader:
        print("Test batch spectrograms shape:", test_batch_x.shape)
        break
    
if __name__ == "__main__":
    main()
