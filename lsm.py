################################################################################
# Title:            lsm.py                                                     #
# Description:      Code to define and train a liquid state machine.           #
# Author:           Aidin Attar                                                #
# Date:             2024-10-02                                                 #
# Version:          0.1                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import os
import torch
import tonic
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import SpykeTorch.functional as sf
import tonic.transforms as transforms
from SpykeTorch import snn
from tqdm import tqdm
from torch.nn.parameter import Parameter
from utils import LatencyEncoding, PoissonSpikeEncoding
import tensorboard as tb

class Reservoir(nn.Module):
    """
    Reservoir layer of the LSM.

    Parameters
    ----------
    input_size : int
        Number of input neurons.
    reservoir_size : int
        Number of reservoir neurons.
    recurrent_prob : float
        Probability of a connection in the reservoir.
    """

    def __init__(self, input_size, reservoir_size, recurrent_prob=0.1):
        super(Reservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        
        # Input-to-reservoir weights
        
        self.input_to_reservoir = snn.Convolution(input_size, reservoir_size, kernel_size=5, weight_mean=0.8, weight_std=0.05)
        
        # Recurrent reservoir connections (randomly connected)
        self.recurrent_connections = nn.Linear(reservoir_size, reservoir_size, bias=False)
        with torch.no_grad():
            mask = torch.rand_like(self.recurrent_connections.weight) < recurrent_prob
            self.recurrent_connections.weight *= mask.float()

        # STDP in the reservoir
        # self.stdp = ReservoirSTDP(self.input_to_reservoir, (0.004, -0.003))
        self.stdp_reservoir = snn.STDP(
            layer=self.input_to_reservoir,
            learning_rate=(0.004, -0.003),
            use_stabilizer=True,
            lower_bound=0,
            upper_bound=1
        )
        self.stdp_recurrent = snn.STDP(
            layer=self.recurrent_connections,
            learning_rate=(0.004, -0.003),
            use_stabilizer=True,
            lower_bound=0,
            upper_bound=1
        )

        self.threshold = 15
        self.n_winners = 5
        self.inhibition_radius = 3
        self.spk_cnt = 0

        self.max_ap = Parameter(torch.tensor([0.15]))

    def forward(self, input_spikes, max_layer = 1):
        # Input-to-reservoir transformation
        pot = self.input_to_reservoir(input_spikes)
        spk, pot = sf.fire(pot, threshold=15, return_thresholded_potentials=True)

        if max_layer == 1:
            self.update_layer_reservoir(input_spikes, pot)
            return spk, pot

        # Flatten the spike output for the linear layer (recurrent connections)
        spk_flat = spk.view(spk.size(0), spk.size(1), -1)  # Flatten (t, channels * height * width)

        # Apply recurrent reservoir dynamics
        pot += self.recurrent_connections(spk_flat.permute(0, 2, 1)).permute(0, 2, 1).view_as(pot)
        spk, pot = sf.fire(pot, threshold=15, return_thresholded_potentials=True)

        self.update_layer_recurrent(input_spikes, pot)
        return spk, pot
    
    def reset(self):
        self.stdp.reset()
        self.input_to_reservoir.reset_state()
        self.recurrent_connections.reset_parameters()

    def update_layer_reservoir(self, input, pot):
        self.spk_cnt += 1
        if self.spk_cnt >= 500:
            self.spk_cnt = 0
            self.update_learning_rate(self.stdp_reservoir)
        pot = sf.pointwise_inhibition(pot)
        spk = pot.sign()
        winners = sf.get_k_winners(pot, self.n_winners, self.inhibition_radius, spk)
        self.update_ctx(input, pot, spk, winners)

    def update_layer_recurrent(self, input, pot):
        self.spk_cnt += 1
        if self.spk_cnt >= 500:
            self.spk_cnt = 0
            self.update_learning_rate(self.stdp_recurrent)
        pot = sf.pointwise_inhibition(pot)
        spk = pot.sign()
        winners = sf.get_k_winners(pot, self.n_winners, self.inhibition_radius, spk)
        self.update_ctx(input, pot, spk, winners)

    def update_ctx(self, input, pot, spk, winners):
        self.ctx = {"input_spikes": input, "potentials": pot, "output_spikes": spk, "winners": winners}

    def update_learning_rate(self, stdp_layer):
        """Update the learning rate of the STDP layer"""
        ap = torch.tensor(stdp_layer.learning_rate[0][0].item(), device=stdp_layer.learning_rate[0][0].device) * 2
        ap = torch.min(ap, self.max_ap)
        an = ap * -0.75
        stdp_layer.update_all_learning_rate(ap.item(), an.item())

    def apply_stdp(self, layer=1):
        if layer == 1:
            self.stdp_reservoir(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        else:
            self.stdp_recurrent(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])


class LSM(nn.Module):
    """
    Liquid State Machine (LSM) model.
    
    Parameters
    ----------
    input_size : int
        Number of input neurons.
    reservoir_size : int
        Number of reservoir neurons.
    output_size : int
        Number of output neurons.
    """
    def __init__(self, input_size, reservoir_size, output_size):
        super(LSM, self).__init__()
        self.reservoir = Reservoir(input_size, reservoir_size)
        self.readout = nn.Linear(reservoir_size, output_size)

    def forward(self, input_spikes):
        reservoir_spikes = self.reservoir(input_spikes)
        return self.readout(reservoir_spikes)
    
    def reset(self):
        self.reservoir.stdp.reset()
        self.reservoir.input_to_reservoir.reset_state()
        self.reservoir.recurrent_connections.reset_parameters()
        self.readout.reset_parameters()

def train_lsm(model, train_loader, optimizer, criterion, epochs=10, device='cuda'):
    """
    Train the LSM model.
    
    Parameters
    ----------
    model : LSM
        LSM model to train.
    train_loader : DataLoader
        DataLoader object for training data.
    optimizer : Optimizer
        Optimizer object for training.
    criterion : Loss
        Loss function for training.
    epochs : int
        Number of training epochs.
    device : str
        Device to train the model on.

    Returns
    -------
    model : LSM
        Trained LSM model.

    Exceptions
    ----------
    None
    """
    model.to(device)
    model.train()

    iterator = tqdm(range(epochs), desc='Training', unit='epoch')
    for epoch in iterator:
        for i, (input_spikes, target) in enumerate(train_loader):
            input_spikes, target = input_spikes.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input_spikes)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=loss.item(), accuracy=(output.argmax(1) == target).float().mean().item())
    return model

def test_lsm(model, test_loader, device='cuda'):
    """
    Test the LSM model.
    
    Parameters
    ----------
    model : LSM
        LSM model to test.
    test_loader : DataLoader
        DataLoader object for testing data.
    device : str
        Device to test the model on.

    Returns
    -------
    accuracy : float
        Accuracy of the model on the test data.

    Exceptions
    ----------
    None
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for input_spikes, target in test_loader:
            input_spikes, target = input_spikes.to(device), target.to(device)
            output = model(input_spikes)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    return accuracy

def train_batch(model, input_spikes, max_layer=1, device='cuda'):
    """
    Train the reservoir layer of the LSM model on a batch of input spikes.

    Parameters
    ----------
    model : LSM
        LSM model to train.
    input_spikes : Tensor
        Batch of input spikes.
    device : str
        Device to train the model on.
    """
    iterator = tqdm(input_spikes, desc='Training', unit='batch', leave=False)
    for data_in in iterator:
        data_in = data_in.to(device)
        model.reservoir(data_in, max_layer=max_layer)
        model.reservoir.apply_stdp(layer=max_layer)

def train_reservoir(model, train_loader, epochs=10, device='cuda'):
    """
    Train the reservoir layer of the LSM model.
    
    Parameters
    ----------
    model : LSM
        LSM model to train.
    train_loader : DataLoader
        DataLoader object for training data.
    epochs : int
        Number of training epochs.
    device : str
        Device to train the model on.

    Returns
    -------
    model : LSM
        Trained LSM model.

    Exceptions
    ----------
    None
    """
    model.to(device)
    model.train()

    iterator = tqdm(range(epochs), desc='Training input layer', unit='epoch')
    for epoch in iterator:
        i = 0
        for i, (input_spikes, _) in enumerate(train_loader):
            input_spikes = input_spikes.to(device)
            train_batch(model, input_spikes, max_layer=1, device=device)
            iterator.set_postfix({"Iteration": i+1})
        
    iterator = tqdm(range(epochs), desc='Training recurrent layer', unit='epoch')
    for epoch in iterator:
        i = 0
        for i, (input_spikes, _) in enumerate(train_loader):
            input_spikes = input_spikes.to(device)
            train_batch(model, input_spikes, max_layer=2, device=device)
            iterator.set_postfix({"Iteration": i+1})
    return model

def main():
    parser = argparse.ArgumentParser(description='Train a Liquid State Machine (LSM) model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use for training.')
    parser.add_argument('--loss', type=str, default='crossentropy', help='Loss function to use for training.')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training.')
    parser.add_argument('--reservoir_size', type=int, default=100, help='Number of reservoir neurons.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on.')
    args = parser.parse_args()

    # Prepare data
    if args.dataset == 'nmnist':
        sensor_size = tonic.datasets.NMNIST.sensor_size
        trans = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=3),
            # transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3),
        ])
        train_dataset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=trans)
        test_dataset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=trans)
    elif args.dataset == 'mnist':
        raise NotImplementedError('MNIST dataset not supported yet.')
    else:
        raise ValueError(f'Dataset {args.dataset} not supported.')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create LSM model
    model = LSM(args.input_size, args.reservoir_size, args.output_size)

    # Train the reservoir layer
    model = train_reservoir(model, train_loader, epochs=args.epochs, device=args.device)

    # Train the full LSM model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model = train_lsm(model, train_loader, optimizer, criterion, epochs=args.epochs, device=args.device)

    # Test the model
    accuracy = test_lsm(model, test_loader, device=args.device)
    print(f'Test accuracy: {accuracy}')