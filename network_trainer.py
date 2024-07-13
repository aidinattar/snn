################################################################################
# Title:            network_trainer.py                                         #
# Description:      Parent class for training neural networks                  #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.3                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging
from torch.utils.tensorboard import SummaryWriter

class NetworkTrainer(nn.Module):
    """Parent class for training neural networks"""

    def __init__(self, num_classes, device="cuda", tensorboard=True):
        super(NetworkTrainer, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.ctx = {
            "input_spikes": None,
            "potentials": None,
            "output_spikes": None,
            "winners": None,
        }
        neurons_per_class = 20
        self.decision_map = [i for i in range(num_classes) for _ in range(neurons_per_class)]
        self.history = {
            'train_acc': [],
            'train_loss': [],
            'test_acc': [],
            'test_loss': []
        }
        self.activation_maps = {}
        self.tensorboard = tensorboard
        if self.tensorboard:
            self.define_writer(f"./runs_{self.__class__.__name__}")
        self.all_preds = []
        self.all_targets = []
        self.iteration = 0
        self.epoch = 0

    def stdp(self):
        """Apply STDP to the network"""
        raise NotImplementedError("This method must be implemented in the child class")

    def update_learning_rates(self):
        """Update learning rates for STDP"""
        raise NotImplementedError("This method should be overridden by subclasses")

    def reward(self):
        """Reward the network"""
        raise NotImplementedError("This method should be overridden by subclasses")

    def punish(self):
        """Punish the network"""
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def train_unsupervised(self, data, layer_idx):
        """
        Train the layer with unsupervised learning (STDP)
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
        layer_idx : int
            Index of the layer to train
        """
        self.train()
        for data_in in data:
            data_in = data_in.to(self.device)
            self(data_in, layer_idx)
            self.stdp(layer_idx)

    def train_rl(self, data, target, layer_idx=3):
        """
        Train the network with reinforcement learning (R-STDP)

        Parameters
        ----------
        data : torch.Tensor
            Input data
        target : torch.Tensor
            Target data
        layer_idx : int
            Index of the layer to train

        Returns
        -------
        perf : np.array
            Performance of the network
        """
        self.train()
        perf = np.array([0, 0, 0])  # correct, wrong, silence
        
        for data_in, target_in in zip(data, target):
            data_in = data_in.to(self.device)
            target_in = target_in.to(self.device)
            d = self(data_in, layer_idx)
            
            if d != -1:
                if d == target_in:
                    perf[0] += 1
                    self.reward()
                else:
                    perf[1] += 1
                    self.punish()
            else:
                perf[2] += 1
        
        avg_loss = perf[1] / (perf[0] + perf[1] + perf[2])
        accuracy = perf[0] / (perf[0] + perf[1] + perf[2])
        
        # self.history['train_loss'].append(avg_loss)
        # self.history['train_acc'].append(accuracy)
        
        # Logging to TensorBoard
        if self.tensorboard:
            self.writer.add_scalar('Train_Iteration/Loss', avg_loss, self.iteration)
            self.writer.add_scalar('Train_Iteration/Accuracy', accuracy, self.iteration)

        self.iteration += 1
        
        return perf / len(data)

    def test(self, data, target, layer_idx=3):
        """
        Test the network
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
        target : torch.Tensor
            Target data
        layer_idx : int
            Index of the layer to test

        Returns
        -------
        perf : np.array
            Performance of the network
        """
        self.eval()
        perf = np.array([0, 0, 0])  # correct, wrong, silence
        for data_in, target_in in zip(data, target):
            data_in = data_in.to(self.device)
            target_in = target_in.to(self.device)
            d = self(data_in, layer_idx)
            if d != -1:
                if d == target_in:
                    perf[0] += 1
                else:
                    perf[1] += 1
            else:
                perf[2] += 1
        return perf / len(data)

    def compute_preds(self, data, target, layer_idx=3):
        """
        Compute evaluation metrics for the network
        
        Parameters
        ----------
        data : torch.Loader
            Input data
        epoch : int
            Current epoch number
        layer_idx : int
            Index of the layer to compute metrics

        Returns
        -------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        self.eval()
        for data_in, target_in in zip(data, target):
            data_in = data_in.to(self.device)
            target_in = target_in.to(self.device)
            d = self(data_in, layer_idx)
            if d != -1:
                self.all_preds.append(d)
                self.all_targets.append(target_in.cpu().item())
        
        self.all_preds = np.array(self.all_preds)
        self.all_targets = np.array(self.all_targets)

    def metrics(self):
        """
        Compute evaluation metrics

        Returns
        -------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        metrics = {
            "confusion_matrix": confusion_matrix(self.all_targets, self.all_preds),
            "f1_score": f1_score(self.all_targets, self.all_preds, average='macro'),
            "accuracy": accuracy_score(self.all_targets, self.all_preds),
            "recall": recall_score(self.all_targets, self.all_preds, average='macro'),
            "precision": precision_score(self.all_targets, self.all_preds, average='macro'),
        }
        return metrics

    def to(self, device):
        """Move the network to the specified device"""
        super(NetworkTrainer, self).to(device)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and all(isinstance(v, nn.Module) for v in attr.values()):
                for key in attr:
                    if isinstance(attr[key], nn.Module):
                        attr[key] = attr[key].to(device)

    def define_writer(self, log_dir):
        """
        Define a SummaryWriter for TensorBoard
        
        Parameters
        ----------
        log_dir : str
            Directory to save TensorBoard logs
        """
        self.tensorboard = True
        self.writer = SummaryWriter(log_dir=log_dir)

    def save_metrics(self, metrics, file_path="metrics.txt"):
        """
        Save evaluation metrics to a file
        
        Parameters
        ----------
        metrics : dict
            Dictionary containing evaluation metrics
        file_path : str
            Path to the file where metrics will be saved
        """
        with open(file_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    def save_history(self, file_path="history.csv"):
        """
        Save training history to a file
        
        Parameters
        ----------
        file_path : str
            Path to the file where history will be saved
        """
        np.savetxt(file_path, np.array([self.history['train_acc'], self.history['train_loss'], self.history['test_acc'], self.history['test_loss']]).T, delimiter=",", header="train_acc,train_loss,test_acc,test_loss")

    def plot_history(self, file_path="history.png"):
        """
        Plot and save training history
        
        Parameters
        ----------
        file_path : str
            Path to the file where plot will be saved
        """
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['test_acc'], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        plt.savefig(file_path)
        plt.show()

    def register_hooks(self):
        """
        Register hooks to save activation maps
        """
        def hook_fn(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(self.activation_maps)
            self.activation_maps[f'{class_name}_{module_idx}'] = output.detach().cpu().numpy()
        
        for layer in self.children():
            layer.register_forward_hook(hook_fn)

    def save_activation_maps(self, file_path="activation_maps"):
        """
        Save activation maps for each layer
        
        Parameters
        ----------
        file_path : str
            Path to the directory where activation maps will be saved
        """
        os.makedirs(file_path, exist_ok=True)
        for name, output in self.activation_maps.items():
            plt.figure(figsize=(10, 10))
            for j in range(output.shape[1]):
                plt.subplot(10, 10, j + 1)
                plt.imshow(output[0, j], cmap='gray')
                plt.axis('off')
            plt.savefig(os.path.join(file_path, f'{name}.png'))
            plt.close()

    def log_tensorboard(self, metrics, epoch):
        """
        Log metrics to TensorBoard
        
        Parameters
        ----------
        metrics : dict
            Dictionary containing evaluation metrics
        epoch : int
            Current epoch number
        """
        for key, value in metrics.items():
            if key == "confusion_matrix":
                fig, ax = plt.subplots()
                cax = ax.matshow(value, cmap='coolwarm')
                fig.colorbar(cax)
                plt.xlabel('True Label')
                plt.ylabel('Predicted Label')
                if self.tensorboard:
                    self.writer.add_figure('Confusion Matrix', fig, epoch)
            else:
                if self.tensorboard:
                    self.writer.add_scalar(key, value, epoch)

    def close_tensorboard(self):
        """
        Close TensorBoard writer
        """
        if self.tensorboard:
            self.writer.close()

    def log_model(self, input_size):
        """
        Log the model graph to TensorBoard

        Parameters
        ----------
        input_size : tuple
            The size of the input tensor
        """
        if not self.tensorboard:
            return
        dummy_input = torch.zeros(1, *input_size).to(self.device)
        self.writer.add_graph(self, dummy_input)
        summary(self, input_size)

    def log_inputs(self, data, epoch, tag='Inputs'):
        """
        Log input images to TensorBoard

        Parameters
        ----------
        data : torch.Tensor
            Input data
        epoch : int
            Current epoch number
        tag : str
            Tag for the input data in TensorBoard
        """
        if not self.tensorboard:
            return
        data = data.to(self.device)
        grid = torchvision.utils.make_grid(data)
        self.writer.add_image(tag, grid, epoch)

    def log_embedding(self, embeddings, metadata, label_img, epoch, tag='Embedding'):
        """
        Log embeddings to TensorBoard

        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings to visualize
        metadata : list
            Metadata corresponding to the embeddings
        label_img : torch.Tensor
            Images corresponding to the embeddings
        epoch : int
            Current epoch number
        tag : str
            Tag for the embeddings in TensorBoard
        """
        if not self.tensorboard:
            return
        self.writer.add_embedding(embeddings, metadata, label_img, global_step=epoch, tag=tag)