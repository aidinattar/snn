################################################################################
# Title:            surrogat_grad.py                                           #
# Description:      This is the training script for the model trained with     #
#                   surrogate gradient                                         #
# Author:           Aidin Attar                                                #
# Date:             2024-09-15                                                 #
# Version:          0.2                                                        #
# Usage:            python surrogat_grad.py                                    #
# Notes:                                                                       #
# Python version:   3.11.7                                                     #
################################################################################

# %%
import os
import torch
import argparse
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import spikeplot as splt
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging
from torch.utils.tensorboard import SummaryWriter

# %%
class SNN(nn.Module):
    """
    Parent class for Spiking Neural Network models.
    """
    def __init__(self):
        super(SNN, self).__init__()

    def forward(self, x):
        pass

    def plot(self, spikes, weights):
        # Plot the spikes and weights of the network
        splt.plot_spikes(spikes)
        splt.plot_weights(weights)

    def save(self, path):
        # Save the model to the specified path
        torch.save(self.state_dict(), path)

    def load(self, path):
        # Load the model from the specified path
        self.load_state_dict(torch.load(path))

    def save_best(self, path, acc):
        # Save the model to the specified path if the accuracy is better than the previous best
        if not os.path.isfile(path) or acc > torch.load(path):
            torch.save(acc, path)

    def metrics(self, test_loader, device, num_steps):
        self.all_targets = []
        self.all_preds = []

        # Iterate over the testing dataset
        for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
            # Move the data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs, _ = forward_pass(self, inputs, num_steps)
            outputs = outputs.sum(0)

            # Compute the predictions
            _, preds = torch.max(outputs, 1)

            # Append the targets and predictions
            self.all_targets.extend(targets.cpu().numpy())
            self.all_preds.extend(preds.cpu().numpy())

        # Compute evaluation metrics
        metrics = {
            "confusion_matrix": confusion_matrix(self.all_targets, self.all_preds),
            "f1_score": f1_score(self.all_targets, self.all_preds, average='macro'),
            "accuracy": accuracy_score(self.all_targets, self.all_preds),
            "recall": recall_score(self.all_targets, self.all_preds, average='macro'),
            "precision": precision_score(self.all_targets, self.all_preds, average='macro'),
        }
        return metrics

    def log_metrics(self, writer, epoch, test_loader, device, num_steps):
        # Log the evaluation metrics
        metrics = self.metrics(test_loader, device, num_steps)
        writer.add_scalar("Metrics/F1Score", metrics["f1_score"], epoch)
        writer.add_scalar("Metrics/Accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("Metrics/Recall", metrics["recall"], epoch)
        writer.add_scalar("Metrics/Precision", metrics["precision"], epoch)

        fig, ax = plt.subplots()
        cax = ax.matshow(metrics["confusion_matrix"], cmap='viridis')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        writer.add_figure("Metrics/ConfusionMatrix", fig, epoch)
        
# %%
class fcSNN(SNN):
    """
    Fully connected Spiking Neural Network model using surrogate gradient.
    """
    def __init__(self, beta, spike_grad, in_features, n_hidden, n_out):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(in_features, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, mem2

# %%
class ConvSNN(SNN):
    """
    Convolutional Spiking Neural Network model using surrogate gradient.
    """
    def __init__(self, beta, spike_grad, kernel_size):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, kernel_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, kernel_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2, 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2, 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(x.size(0), -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3

# %%
def forward_pass(model, data, num_steps):
    """
    Perform a forward pass through the network.
    """
    mem_rec = []
    spk_rec = []
    
    # Reset the internal states of the model
    utils.reset(model)

    for _ in range(num_steps):
        spk_out, mem_out = model(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

# %%
def batch_accuracy(train_loader, net, num_steps, device):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            spk_rec, _ = forward_pass(net, data, num_steps)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total

# %%
def load_data(batch_size, dataset):
    """
    Load the specified dataset using PyTorch DataLoader.
    """
    # Define the transformations to apply to the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])

    # Load the training and testing datasets
    if dataset == "mnist":
        train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    elif dataset == "fashion":
        train_dataset = datasets.FashionMNIST(root="data", train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root="data", train=False, transform=transform, download=True)
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
    else:
        raise ValueError("Invalid dataset specified. Please use 'mnist', 'fashion', or 'cifar10'.")

    # Create the DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

# %%
def train(model, train_loader, optimizer, loss_fn, num_steps, device, writer = None, epoch = 0):
    """
    Train the model on the training dataset.
    """
    # Set the model to training mode
    model.train()

    # Initialize the loss and accuracy
    total_loss = 0.0
    total_correct = 0

    batch_idx = 0
    counter = 0

    # Iterate over the training dataset
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        # Move the data to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs, _ = forward_pass(model, inputs, num_steps)
        loss = loss_fn(outputs, targets)

        # print("Accuracy: ", batch_accuracy(train_loader, model, num_steps, device))

        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute the loss and accuracy
        batch_loss = loss.item()
        batch_correct = SF.accuracy_rate(outputs, targets) * outputs.size(1)
        total_loss += batch_loss
        total_correct += batch_correct

        # Log the loss and accuracy
        # print(f"Loss: {batch_loss:.4f}, Accuracy: {batch_correct / inputs.size(0):.4f}")
        if writer is not None:
            writer.add_scalar("Loss/TrainBatch", batch_loss, len(train_loader) * epoch + batch_idx)
            writer.add_scalar("Accuracy/TrainBatch", batch_correct / inputs.size(0), len(train_loader) * epoch + batch_idx)
        batch_idx += 1

    return total_loss / len(train_loader), total_correct / len(train_loader.dataset)

# %%
def test(model, test_loader, loss_fn, num_steps, device, writer = None, epoch = 0):
    """
    Test the model on the testing dataset.
    """
    # Set the model to evaluation
    model.eval()

    # Initialize the loss and accuracy
    total_loss = 0.0
    total_correct = 0

    # Disable gradient computation
    with torch.no_grad():
        batch_idx = 0
        # Iterate over the testing dataset
        for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
            # Move the data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs, _ = forward_pass(model, inputs, num_steps)
            loss = loss_fn(outputs, targets)

            # Compute the loss and accuracy
            batch_loss = loss.item()
            batch_correct = SF.accuracy_rate(outputs, targets) * outputs.size(1)
            total_loss += batch_loss
            total_correct += batch_correct
            
            # Log the loss and accuracy
            # print(f"Loss: {batch_loss:.4f}, Accuracy: {batch_correct / inputs.size(0):.4f}")
            if writer is not None:
                writer.add_scalar("Loss/TestBatch", batch_loss, len(test_loader) * epoch + batch_idx)
                writer.add_scalar("Accuracy/TestBatch", batch_correct / inputs.size(0), len(test_loader) * epoch + batch_idx)

            batch_idx += 1

    return total_loss / len(test_loader), total_correct / len(test_loader.dataset)

# %%
def plot_spikes_weights(model, device, test_loader):
    """
    Plot the spikes and weights of the network using snnTorch.
    """
    model.eval()
    spikes = []
    weights = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Plotting", leave=False):
            inputs = inputs.to(device)
            
            # Forward pass through the model
            spk4 = model(inputs)
            
            # Collect spikes
            spikes.append([
                model.lif1.mem.cpu().numpy(),
                model.lif2.mem.cpu().numpy(),
                model.lif3.mem.cpu().numpy(),
            ])

            # Collect weights
            weights.append([
                model.conv1.weight.cpu().numpy(),
                model.conv2.weight.cpu().numpy(),
                model.fc1.weight.cpu().numpy(),
            ])

    # Plot spikes using snnTorch's spikeplot utilities
    splt.plot_spikes(spikes)
    plt.show()

    # Plot weights manually using matplotlib
    plt.figure(figsize=(10, 5))
    for idx, weight in enumerate(weights[0]):
        plt.subplot(1, len(weights[0]), idx + 1)
        plt.imshow(weight)
        plt.title(f"Layer {idx + 1} Weights")
    plt.show()

# %%
def main():
    # Define the parser
    parser = argparse.ArgumentParser(description="Train a spiking neural network on the MNIST dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use for training.")
    parser.add_argument("--loss_fn", type=str, default="crossentropy", help="Loss function to use for training.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer.")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta value for the LIF neuron.")
    parser.add_argument("--spike_grad", type=str, default="fast_sigmoid", help="Spike gradient to use for training.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the network.")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for the convolutional layers.")
    parser.add_argument("--n_filters", type=int, default=32, help="Number of filters for the convolutional layers.")
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in the fully connected layer.")
    parser.add_argument("--n_out", type=int, default=10, help="Number of output units in the fully connected layer.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--log_dir", type=str, default="runs_surrogate", help="Directory to save logs to.")
    parser.add_argument("--save_dir", type=str, default="saves", help="Directory to save model checkpoints to.")
    parser.add_argument("--plot", action="store_true", help="Plot the spikes and weights of the network.")
    parser.add_argument("--metrics", action="store_true", help="Compute the metrics at the end of the epoch.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use for reproducibility.")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use for training.")
    parser.add_argument("--model", type=str, default="fcsnn", help="Model to use for training.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of simulation steps.")
    parser.add_argument("--tensorboard", action="store_true", help="Use TensorBoard for logging.")
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Print the command line arguments
    print("############################################")
    print("Program to train the model")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print("############################################")

    # %%
    if not os.path.isdir("models"):
        os.mkdir("models")

    # Load the data
    train_loader, test_loader = load_data(args.batch_size, args.dataset)

    # %%
    if args.spike_grad == "fast_sigmoid":
        spike_grad = surrogate.fast_sigmoid(slope=25)
    elif args.spike_grad == "atan":
        spike_grad = surrogate.atan()
    elif args.spike_grad == "heaviside":
        spike_grad = surrogate.heaviside()
    else:
        raise ValueError("Invalid spike gradient specified. Please use 'fast_sigmoid', 'atan', or 'heaviside'.")

    # %%
    # Define the network
    if args.model == "fcsnn":
        model = fcSNN(args.beta, spike_grad, 28*28, args.n_hidden, args.n_out)
    elif args.model == "convsnn":
        model = ConvSNN(args.beta, spike_grad, args.kernel_size)
    else:
        raise ValueError("Invalid model specified. Please use 'snn'.")
    
    # %%
    # Move the model to the specified device
    model.to(args.device)

    # %%
    # Define the optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Invalid optimizer specified. Please use 'adam' or 'sgd'.")
    
    # %%
    # Define the loss function
    if args.loss_fn == "crossentropy":
        loss_fn = SF.ce_rate_loss()
    else:
        raise ValueError("Invalid loss function specified. Please use 'crossentropy'.")
    
    # %%
    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Initialize the best accuracy
    best_acc = 0.0

    # Initialize the TensorBoard writer
    if args.tensorboard:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    # Log the model graph
    # scripted_model = torch.jit.script(model)
    # writer.add_graph(scripted_model, next(iter(train_loader))[0].to(args.device))

    # %%
    # Train the model
    iterator = tqdm(range(args.epochs), desc="Epoch")
    for epoch in iterator:
        # Train the model
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, args.num_steps, args.device, writer, epoch)

        # Test the model
        test_loss, test_acc = test(model, test_loader, loss_fn, args.num_steps, args.device, writer, epoch)

        # Compute the metrics and log them in tensorboard
        if args.metrics and writer is not None:
            model.log_metrics(writer, epoch, test_loader, args.device, args.num_steps)

        # Update the learning rate
        scheduler.step(test_loss)

        # Print the results
        # print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        iterator.set_postfix(train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc)

        # Log the results
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)

        # Save the model if the accuracy is better
        if test_acc > best_acc:
            best_acc = test_acc
            model.save_best(f"models/surrogate_{args.model}_{args.dataset}.pt", best_acc)

    # Plot the spikes and weights of the network
    if args.plot:
        plot_spikes_weights(model, args.device, test_loader)

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()