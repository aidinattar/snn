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
import SpykeTorch as snn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import SpykeTorch.functional as sf
import tonic.transforms as transforms
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
        self.input_to_reservoir = snn.Convolution(input_size, reservoir_size, kernel_size=5, weight_decay=0.05)
        
        # Recurrent reservoir connections (randomly connected)
        self.recurrent_connections = nn.Linear(reservoir_size, reservoir_size, bias=False)
        with torch.no_grad():
            mask = torch.rand_like(self.recurrent_connections.weight) < recurrent_prob
            self.recurrent_connections.weight *= mask.float()

        # STDP in the reservoir
        self.stdp = ReservoirSTDP(self.reservoir_layer, (0.004, -0.003))

    def forward(self, input_spikes):
        # Input-to-reservoir transformation
        pot = self.input_to_reservoir(input_spikes)
        spk, pot = sf.fire(pot, threshold=15)
        
        # Apply recurrent reservoir dynamics
        pot += self.recurrent_connections(spk)
        spk, pot = sf.fire(pot, threshold=15)

        return spk
    
    def reset(self):
        self.stdp.reset()
        self.input_to_reservoir.reset_state()
        self.recurrent_connections.reset_parameters()


class ReservoirSTDP(nn.Module):
    def __init__(self, layer, learning_rate, use_stabilizer=True, lower_bound=0, upper_bound=1):
        super(ReservoirSTDP, self).__init__()
        self.layer = layer
        if isinstance(learning_rate, list):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = [learning_rate] * layer.out_features
        for i in range(layer.out_features):
            self.learning_rate[i] = (Parameter(torch.tensor([self.learning_rate[i][0]])),
                                     Parameter(torch.tensor([self.learning_rate[i][1]])))
            self.register_parameter('ltp_' + str(i), self.learning_rate[i][0])
            self.register_parameter('ltd_' + str(i), self.learning_rate[i][1])
            self.learning_rate[i][0].requires_grad_(False)
            self.learning_rate[i][1].requires_grad_(False)
        self.use_stabilizer = use_stabilizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, input_spikes, output_spikes):
        # Compute spike timing differences
        spike_timing_diff = input_spikes - output_spikes  # Approximation
        
        # Get the sign of the spike timing to apply LTP or LTD
        ltp_mask = spike_timing_diff <= 0  # LTP: Pre before post (or concurrent)
        ltd_mask = spike_timing_diff > 0   # LTD: Post before pre
        
        # Apply weight updates based on spike timing
        lr = torch.zeros_like(self.layer.weight)
        for i in range(len(self.layer.weight)):
            lr[i] = torch.where(ltp_mask, self.learning_rate[i][0], self.learning_rate[i][1])

        # Update weights with stabilizer term
        self.layer.weight += lr * (
            (self.layer.weight - self.lower_bound) * (self.upper_bound - self.layer.weight) if self.use_stabilizer else 1
        )
        self.layer.weight.clamp_(self.lower_bound, self.upper_bound)


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

    iterator = tqdm(range(epochs), desc='Training', unit='epoch')
    for epoch in iterator:
        for i, (input_spikes, _) in enumerate(train_loader):
            input_spikes = input_spikes.to(device)
            output = model.reservoir(input_spikes)
            model.reservoir.stdp(input_spikes, output)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train a Liquid State Machine (LSM) model.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use for training.')
    parser.add_argument('--loss', type=str, default='crossentropy', help='Loss function to use for training.')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training.')
    parser.add_argument('--reservoir-size', type=int, default=100, help='Number of reservoir neurons.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on.')
    args = parser.parse_args()

    # Prepare data
    if args.dataset == 'nmnist':
        

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