################################################################################
# Title:            lsm.py                                                     #
# Description:      Code to define and train a liquid state machine.           #
# Author:           Aidin Attar                                                #
# Date:             2024-10-02                                                 #
# Version:          0.2                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import os
import argparse
import tonic
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import snntorch as snn
from torch import optim

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging

# Define the early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def initialize_weights(input_weight_scale, reservoir_weight_scale, input_connection_density, input_size, decay_factor=9, inhibitory_fraction=0.2, grid_x=10, grid_y=10, grid_z=10, initialize_reservoir_weights=True, reservoir_weights=None):
    """
    Initializes input-to-reservoir and reservoir-to-reservoir weights for an LSM.

    Parameters
    ---------
    input_weight_scale : float
        Scaling factor for input-to-reservoir connections.
    reservoir_weight_scale : float
        Scaling factor for reservoir-to-reservoir connections.
    input_connection_density : float
        Fraction of reservoir neurons connected to each input neuron.
    input_size : int
        Number of input neurons.
    decay_factor : float
        Spatial decay factor for reservoir connections.
    inhibitory_fraction : float
        Fraction of neurons in the reservoir that are inhibitory.
    grid_x : int
        Size of the reservoir grid in the x-dimension.
    grid_y : int
        Size of the reservoir grid in the y-dimension.
    grid_z : int
        Size of the reservoir grid in the z-dimension.
    initialize_reservoir_weights : bool
        Whether to initialize reservoir weights.
    reservoir_weights : np.ndarray or None
        Pre-initialized reservoir weights, if available.

    Returns
    -------
    Tuple
    - input_weights (np.ndarray): Input-to-reservoir weight matrix.
    - reservoir_weights (np.ndarray): Reservoir-to-reservoir weight matrix.
    """
    num_reservoir_neurons = grid_x * grid_y * grid_z
    input_weights = np.zeros((input_size, num_reservoir_neurons))
    num_input_connections = np.int32(num_reservoir_neurons * input_connection_density)

    # Initialize input-to-reservoir weights
    for input_neuron in range(input_size):
        neuron_indices = np.arange(num_reservoir_neurons)
        np.random.shuffle(neuron_indices)
        positive_connections = neuron_indices[:num_input_connections]
        negative_connections = neuron_indices[-num_input_connections:]

        input_weights[input_neuron, positive_connections] = input_weight_scale
        input_weights[input_neuron, negative_connections] = -input_weight_scale

    # Reservoir neuron inhibition/excitation setup
    neuron_indices = np.arange(num_reservoir_neurons)
    np.random.shuffle(neuron_indices)
    inhibitory_count = np.int32(inhibitory_fraction * num_reservoir_neurons)

    # Initialize reservoir-to-reservoir weights (if needed)
    if initialize_reservoir_weights:
        reservoir_weights = np.zeros((num_reservoir_neurons, num_reservoir_neurons))

        iterator = tqdm(range(num_reservoir_neurons), desc='Initializing reservoir weights', unit='neuron', position=0, leave=True, dynamic_ncols=True, total=num_reservoir_neurons, initial=0, ascii=True)
        for post_idx in iterator:
            post_neuron = neuron_indices[post_idx]
            z_post = post_neuron // (grid_x * grid_y)
            y_post = (post_neuron - z_post * grid_x * grid_y) // grid_x
            x_post = post_neuron % grid_x

            for pre_idx in range(num_reservoir_neurons):
                pre_neuron = neuron_indices[pre_idx]
                z_pre = pre_neuron // (grid_x * grid_y)
                y_pre = (pre_neuron - z_pre * grid_x * grid_y) // grid_x
                x_pre = pre_neuron % grid_x

                distance_squared = (x_post - x_pre) ** 2 + (y_post - y_pre) ** 2 + (z_post - z_pre) ** 2
                probability = np.exp(-distance_squared / decay_factor)

                if post_idx < inhibitory_count and pre_idx < inhibitory_count:
                    # Inhibitory-to-inhibitory connection (II)
                    connection_prob = 0.3 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = -reservoir_weight_scale
                elif post_idx < inhibitory_count and pre_idx >= inhibitory_count:
                    # Excitatory-to-inhibitory connection (EI)
                    connection_prob = 0.1 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = reservoir_weight_scale
                elif post_idx >= inhibitory_count and pre_idx < inhibitory_count:
                    # Inhibitory-to-excitatory connection (IE)
                    connection_prob = 0.05 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = -reservoir_weight_scale
                else:
                    # Excitatory-to-excitatory connection (EE)
                    connection_prob = 0.2 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = reservoir_weight_scale

        # Ensure no self-connections
        np.fill_diagonal(reservoir_weights, 0)

    return input_weights.T, reservoir_weights.T  # Transposed for compatibility with PyTorch nn.Linear

def initialize_partitioned_weights(input_weight_scale, local_reservoir_weight_scale, long_distance_inhibitory_weight_scale, input_connection_density, input_size, num_partitions, decay_factor=9, inhibitory_fraction=0.2, grid_x=10, grid_y=10, grid_z=10, initialize_reservoir_weights=True, reservoir_weights=None):
    """
    Initializes input-to-reservoir and reservoir-to-reservoir weights for a partitioned LSM with cross-partition inhibition.

    Parameters
    ----------
    input_weight_scale : float
        Scaling factor for input-to-reservoir connections.
    local_reservoir_weight_scale : float
        Scaling factor for reservoir-to-reservoir local connections.
    long_distance_inhibitory_weight_scale : float
        Scaling factor for cross-partition inhibition connections.
    input_connection_density : float
        Fraction of reservoir neurons connected to each input neuron.
    input_size : int
        Number of input neurons.
    num_partitions : int
        Number of partitions for the reservoir.
    decay_factor : float
        Spatial decay factor for reservoir connections.
    inhibitory_fraction : float
        Fraction of neurons in the reservoir that are inhibitory.
    grid_x : int
        Size of the reservoir grid in the x-dimension.
    grid_y : int
        Size of the reservoir grid in the y-dimension.
    grid_z : int
        Size of the reservoir grid in the z-dimension.
    initialize_reservoir_weights : bool
        Whether to initialize reservoir weights.
    reservoir_weights : np.ndarray or None
        Pre-initialized reservoir weights, if available.

    Returns
    -------
    Tuple
    - input_weights (list of np.ndarray): Input-to-reservoir weight matrices for each partition.
    - reservoir_weights (np.ndarray): Reservoir-to-reservoir local weight matrix.
    - long_distance_inhibitory_weights (np.ndarray): Cross-partition long-distance inhibitory connections.
    """
    partition_size_z = grid_z // num_partitions
    total_neurons = grid_x * grid_y * grid_z
    partition_neurons = grid_x * grid_y * partition_size_z

    input_connection_range = np.int32(partition_neurons * input_connection_density)
    input_weights_partition = np.zeros((input_size, partition_neurons))
    input_weights = []

    # Initialize input-to-reservoir weights for each partition
    for input_neuron in range(input_size):
        neuron_indices = np.arange(partition_neurons)
        np.random.shuffle(neuron_indices)
        positive_connections = neuron_indices[:input_connection_range]
        negative_connections = neuron_indices[-input_connection_range:]

        input_weights_partition[input_neuron, positive_connections] = input_weight_scale
        input_weights_partition[input_neuron, negative_connections] = -input_weight_scale

    # Calculate statistics
    active_connections = input_weights_partition > 0
    input_fanout = np.sum(active_connections, axis=1) / input_weight_scale
    reservoir_fanin = np.sum(active_connections, axis=0) / input_weight_scale
    print(f'Input-to-reservoir fanout: {np.mean(input_fanout):.2f}, shape: {input_fanout.shape}')
    print(f'Reservoir fanin from input: {np.mean(reservoir_fanin):.2f}, shape: {reservoir_fanin.shape}')

    # Assign input weights to different partitions
    for partition in range(num_partitions):
        partition_input_weights = np.zeros((input_size, total_neurons))
        partition_input_weights[:, partition * partition_neurons:(partition + 1) * partition_neurons] = input_weights_partition
        input_weights.append(partition_input_weights.T)

    # Initialize reservoir weights (local connections) and cross-partition inhibitory weights
    neuron_indices = np.arange(partition_neurons)
    np.random.shuffle(neuron_indices)
    inhibitory_count = np.int32(inhibitory_fraction * partition_neurons)

    if initialize_reservoir_weights:
        local_reservoir_weights = np.zeros((total_neurons, total_neurons))
        cross_partition_inhibitory_weights = np.zeros((total_neurons, total_neurons))

        for i in range(partition_neurons):
            post_neuron = neuron_indices[i]
            z_post = post_neuron // (grid_x * grid_y)
            y_post = (post_neuron - z_post * grid_x * grid_y) // grid_x
            x_post = post_neuron % grid_x

            for j in range(partition_neurons):
                pre_neuron = neuron_indices[j]
                z_pre = pre_neuron // (grid_x * grid_y)
                y_pre = (pre_neuron - z_pre * grid_x * grid_y) // grid_x
                x_pre = pre_neuron % grid_x

                distance_squared = (x_post - x_pre) ** 2 + (y_post - y_pre) ** 2 + (z_post - z_pre) ** 2
                connection_probability = np.exp(-distance_squared / decay_factor)

                # Determine connection type and assign weights
                if i < inhibitory_count and j < inhibitory_count:
                    # Inhibitory-to-inhibitory (II) connections
                    if np.random.uniform() < 0.3 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = -local_reservoir_weight_scale
                elif i < inhibitory_count and j >= inhibitory_count:
                    # Excitatory-to-inhibitory (EI) connections
                    if np.random.uniform() < 0.1 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = local_reservoir_weight_scale
                elif i >= inhibitory_count and j < inhibitory_count:
                    # Inhibitory-to-excitatory (IE) connections
                    if np.random.uniform() < 0.05 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = -local_reservoir_weight_scale
                else:
                    # Excitatory-to-excitatory (EE) connections
                    if np.random.uniform() < 0.2 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = local_reservoir_weight_scale

        # Ensure no self-connections in the local reservoir
        np.fill_diagonal(local_reservoir_weights, 0)

        # Create cross-partition inhibitory connections
        for i in range(total_neurons):
            cross_partition_inhibitory_weights[i, (i + partition_neurons) % total_neurons] = -long_distance_inhibitory_weight_scale

        # Assign local weights to each partition
        for partition in range(num_partitions):
            local_reservoir_weights[partition * partition_neurons:(partition + 1) * partition_neurons,
                                    partition * partition_neurons:(partition + 1) * partition_neurons] = local_reservoir_weights[:partition_neurons, :partition_neurons]

    return input_weights, local_reservoir_weights.T, cross_partition_inhibitory_weights.T

class LSM(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        n_partitions: int=3,
        weight_in: np.ndarray = None,
        weight_res: np.ndarray = None,
        sparsity: float=0.1,
        alpha: float=0.9,
        beta: float=0.9,
        threshold: float=1.0
    ):
        super(LSM, self).__init__()

        # Input layer (fully connected)
        self.input_fc = nn.Linear(input_size, reservoir_size)
        if weight_in is not None:
            self.input_fc.weight = nn.Parameter(torch.from_numpy(weight_in))

        # Define the reservoir layer
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.n_partitions = n_partitions

        # Synaptic integration dynamics
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=reservoir_size, threshold=threshold)
        if weight_res is not None:
            self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(weight_res))
        
        # Output (readout) layer
        self.readout = nn.Linear(reservoir_size * n_partitions, output_size)

        # Freeze the weights of the input and reservoir layers
        self._freeze_weights()

        # Convert the model to double precision
        # self.double()
        self.to(torch.float32)

    def forward(self, x):
        # x = x.double()
        # Flatten the input data
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))

        num_steps = x.size(0)  # Number of time steps (assuming spike data is already encoded)
        spk, syn, mem = self.lsm.init_rsynaptic()  # Initialize membrane potential and synapse states
        
        spk_rec = []

        for step in range(num_steps):
            # Input to reservoir transformation
            current_input = self.input_fc(x[step])
            
            # Reservoir update
            spk, syn, mem = self.lsm(current_input, spk, syn, mem)

            spk_rec.append(spk)

        # Stack spikes over time
        spk_rec_out = torch.stack(spk_rec)

        # Partition the reservoir output
        partition_steps = num_steps // self.n_partitions
        lsm_parts = []

        # Aggregate spikes over each partition
        for partition in range(self.n_partitions):
            lsm_parts.append(torch.mean(spk_rec_out[partition * partition_steps:(partition + 1) * partition_steps], dim=0))
        readout_input = torch.cat(lsm_parts, dim=1)

        # Pool the spike responses from the reservoir to pass to the readout layer
        output = self.readout(readout_input)

        return output, spk_rec_out
    
    def _freeze_weights(self):
        """
        Freezes the weights of the input and reservoir layers.
        """
        for param in self.input_fc.parameters():
            param.requires_grad = False
        for param in self.lsm.recurrent.parameters():
            param.requires_grad = False

class LSM_partition(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        weight_in: np.ndarray=None,
        weight_lin: np.ndarray=None,
        weight_res: np.ndarray=None,
        n_partitions: int=3,
        sparsity: float=0.1,
        alpha: float=0.9,
        beta: float=0.9,
        threshold: float=1.0
    ):
        super(LSM_partition, self).__init__()

        # Input layer (fully connected)
        self.input_fc = nn.Linear(input_size, reservoir_size)
        if weight_in is not None:
            self.input_fc.weight = nn.Parameter(torch.from_numpy(weight_in))

        # Define the reservoir layer
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.n_partitions = n_partitions
        self.weight_in = weight_in

        # Initialize random sparse connectivity for the reservoir
        self.reservoir = nn.Linear(reservoir_size, reservoir_size, bias=False)
        if weight_lin is not None:
            self.reservoir.weight = nn.Parameter(torch.from_numpy(weight_lin))
        else:
            self.init_reservoir_connections()

        # Synaptic integration dynamics
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=reservoir_size, threshold=threshold)
        if weight_res is not None:
            self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(weight_res))

        # Output (readout) layer
        self.readout = nn.Linear(reservoir_size, output_size)

        # Freeze the weights of the input and reservoir layers
        self._freeze_weights()

    def _freeze_weights(self):
        """
        Freezes the weights of the input and reservoir layers.
        """
        for param in self.input_fc.parameters():
            param.requires_grad = False
        for param in self.reservoir.parameters():
            param.requires_grad = False
        for param in self.lsm.recurrent.parameters():
            param.requires_grad = False

    def init_reservoir_connections(self):
        # Randomly mask connections based on sparsity
        with torch.no_grad():
            mask = torch.rand_like(self.reservoir.weight) < self.sparsity
            self.reservoir.weight *= mask.float()

    def forward(self, x):
        # Flatten the input data
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))

        num_steps = x.size(0)  # Number of time steps (assuming spike data is already encoded)
        spk, syn, mem = self.lsm.init_rsynaptic()  # Initialize membrane potential and synapse states
        
        spk_rec = []

        partition_steps = num_steps // self.n_partitions
        partition_idx = 0

        for step in range(num_steps):
            if step % partition_steps == 0:
                self.input_fc.weight = nn.Parameter(torch.from_numpy(self.weight_in[partition_idx]))
                partition_idx = (partition_idx + 1) % self.n_partitions

            # Input to reservoir transformation
            current_input = self.input_fc(x[step])
            if step > partition_steps:
                current_input += self.reservoir(spk_rec[step - partition_steps])

            spk, syn, mem = self.lsm(current_input, spk, syn, mem)

            spk_rec.append(spk)

        # Stack spikes over time
        spk_rec_out = torch.stack(spk_rec)

        # Pool the spike responses from the reservoir to pass to the readout layer
        readout_input = torch.mean(spk_rec_out, dim=0)  # Aggregate over time
        output = self.readout(readout_input)

        return output, spk_rec_out


def main():
    parser = argparse.ArgumentParser(description='Train a Liquid State Machine (LSM) model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use for training.')
    parser.add_argument('--loss', type=str, default='crossentropy', help='Loss function to use for training.')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training.')
    parser.add_argument('--reservoir_size', type=int, default=1000, help='Reservoir size for the LSM.')
    parser.add_argument('--output_size', type=int, default=10, help='Output size for the LSM.')
    parser.add_argument('--num_partitions', type=int, default=3, help='Number of partitions for the LSM.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--sparsity', type=float, default=0.1, help='Sparsity of the reservoir connections.')
    parser.add_argument('--alpha', type=float, default=0.9, help='Membrane potential decay rate.')
    parser.add_argument('--beta', type=float, default=0.9, help='Synaptic decay rate.')
    parser.add_argument('--threshold', type=float, default=1.0, help='Threshold for the LSM.')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging.')
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f'runs_LSM/{args.dataset}')

    # Prepare data
    if args.dataset == 'nmnist':
        sensor_size = tonic.datasets.NMNIST.sensor_size
        trans = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=3000),
            # tonic.transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=100),
            tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ])
        train_dataset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=trans)
        test_dataset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=trans)
    elif args.dataset == 'mnist':
        raise NotImplementedError('MNIST dataset not supported yet.')
    else:
        raise ValueError(f'Dataset {args.dataset} not supported.')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False))

    # Set neuron parameters
    tauV = 16.0
    tauI = 16.0
    threshold = 20
    current_prefactor = np.float32(1/tauI)
    alpha = np.float32(np.exp(-1/tauI))
    beta = np.float32(1 - 1/tauV)

    data, targets = next(iter(train_loader))
    flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1))

    # Log the dataset in TensorBoard
    # if args.tensorboard:
    #     writer.add_embedding(flat_data, metadata=targets, label_img=data)

    # Set the input size
    input_shape = flat_data.shape[-1]

    # Initialize random sparse connectivity for the reservoir
    input_weights, reservoir_weights = initialize_weights(
        input_weight_scale=27,
        reservoir_weight_scale=2,
        input_connection_density=0.15,
        input_size=input_shape,    
    )

    input_weights *= current_prefactor
    reservoir_weights *= current_prefactor

    model = LSM(
        input_shape,
        args.reservoir_size,
        args.output_size,
        n_partitions=args.num_partitions,
        weight_in=input_weights,
        weight_res=reservoir_weights,
        sparsity=args.sparsity,
        alpha=alpha,
        beta=beta,
        threshold=threshold
    ).to(args.device)

    # Log the model in TensorBoard
    # if args.tensorboard:
    #     writer.add_graph(model, data.to(args.device))

    # Define the loss function
    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'bce':
        criterion = nn.BCELoss()
    else:
        raise ValueError(f'Loss function {args.loss} not supported.')

    # Define the optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError(f'Optimizer {args.optimizer} not supported.')

    # Print model summary
    print(model)

    # Define scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Define early stopping
    early_stopping = EarlyStopping(patience=5, verbose=False)

    # Training loop
    iterator = tqdm(range(args.epochs), desc='Training', unit='epoch', position=0, leave=True, dynamic_ncols=True, total=args.epochs, initial=0, ascii=True)
    for epoch in iterator:
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        iterator_epoch = tqdm(train_loader, desc=f'Epoch {epoch + 1}', unit='batch', position=1, leave=False, dynamic_ncols=True, total=len(train_loader), initial=0, ascii=True)
        for i, (data, target) in enumerate(iterator_epoch):
            data, target = data.to(args.device), target.to(args.device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output, _ = model(data)

            # Compute the loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy_score(target.cpu(), output.argmax(dim=1).cpu())

            # Log the loss and accuracy
            iterator_epoch.set_postfix(loss=loss.item(), accuracy=accuracy_score(target.cpu(), output.argmax(dim=1).cpu()))

            # Log the loss and accuracy in TensorBoard
            if args.tensorboard:
                writer.add_scalar('Loss_iter/train', loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Accuracy_iter/train', accuracy_score(target.cpu(), output.argmax(dim=1).cpu()), epoch * len(train_loader) + i)

        iterator.set_postfix(loss=epoch_loss, accuracy=epoch_accuracy/(i+1))
        if args.tensorboard:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy/(i+1), epoch)

        # Update the learning rate and check for early stopping
        scheduler.step()
        early_stopping(epoch_loss, model)

    # Evaluation loop
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        iterator = tqdm(test_loader, desc='Evaluation', unit='batch', position=0, leave=True, dynamic_ncols=True, total=len(test_loader), initial=0, ascii=True)
        for i, (data, target) in enumerate(iterator):
            data, target = data.to(args.device), target.to(args.device)

            # Forward pass
            output, _ = model(data)

            # Log the predictions
            y_true.extend(target.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())

            iterator.set_postfix(accuracy=accuracy_score(y_true, y_pred))

        # Log the accuracy and confusion matrix in TensorBoard
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        if args.tensorboard:
            writer.add_scalar('Accuracy/test', accuracy, 0)
            writer.add_image('Confusion Matrix', cm, 0)

    # Compute the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test accuracy: {accuracy}')

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()