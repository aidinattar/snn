#################################################################################
# Reimplementation of the 10-Class Digit Recognition Experiment Performed in:   #
# https://arxiv.org/abs/1804.00227                                              #
#                                                                               #
# Reference:                                                                    #
# Mozafari, Milad, et al.,                                                      #
# "Combining STDP and Reward-Modulated STDP in                                  #
# Deep Convolutional Spiking Neural Networks for Digit Recognition."            #
# arXiv preprint arXiv:1804.00227 (2018).                                       #
#                                                                               #
# Original implementation (in C++/CUDA) is available upon request.              #
#################################################################################

import os
import matplotlib.pyplot as plt

plt.ion()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

use_cuda = True
train = True

class MozafariMNIST2018(nn.Module):
    def __init__(self):
        super(MozafariMNIST2018, self).__init__()

        # Layer 1
        # Convolutional Layer
        self.conv1 = snn.Convolution(6, 30, 5, 0.8, 0.05)
        # Time to live for the spikes
        self.conv1_t = 15
        # Number of winners
        self.k1 = 5
        # Radius of the inhibition
        self.r1 = 3

        self.conv2 = snn.Convolution(30, 250, 3, 0.8, 0.05)
        self.conv2_t = 10
        self.k2 = 8
        self.r2 = 1

        self.conv3 = snn.Convolution(250, 200, 5, 0.8, 0.05)

        # STDP
        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
        # Anti-STDP
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
        # Adaptive learning rates
        self.max_ap = Parameter(torch.Tensor([0.15]))

        # Decision Map, 20 neurons for each digit
        self.decision_map = []
        for i in range(10):
            self.decision_map.extend([i]*20)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input, max_layer):
        # Padding the input to avoid edge effects
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            # Layer 1
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.ctx["input_spikes"] = input
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self.ctx["input_spikes"] = spk_in
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2))
            pot = self.conv3(spk_in)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def train_rl(network, data, target):
    """
    Train the network using reinforcement learning
    
    Parameters
    ----------
    network : MozafariMNIST2018
        The network to train
    data : torch.Tensor
        The input data
    target : torch.Tensor
        The target data

    Returns
    -------
    numpy.ndarray
        The performance of the network
    """
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)

def test(network, data, target):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)

class S1C1Transform:
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt+=1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()

kernels = [ utils.DoGKernel(3,3/9,6/9),
            utils.DoGKernel(3,6/9,3/9),
            utils.DoGKernel(7,7/9,14/9),
            utils.DoGKernel(7,14/9,7/9),
            utils.DoGKernel(13,13/9,26/9),
            utils.DoGKernel(13,26/9,13/9)]
filter = utils.Filter(kernels, padding = 6, thresholds = 50)
s1c1 = S1C1Transform(filter)

data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=False)
MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

mozafari = MozafariMNIST2018()
if use_cuda:
    mozafari.cuda()

# Save accuracy and loss for each epoch
train_acc = []
train_loss = []
test_acc = []
test_loss = []

# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1.net"):
    print("Loading the saved network")
    mozafari.load_state_dict(torch.load("saved_l1.net"))
else:
    for epoch in range(2):
        print("Epoch", epoch)
        iter = 0
        for data,targets in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 1)
            print("Done!")
            iter+=1
    torch.save(mozafari.state_dict(), "saved_l1.net")
# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2.net"):
    print("Loading the saved network")
    mozafari.load_state_dict(torch.load("saved_l2.net"))
else:
    for epoch in range(4):
        print("Epoch", epoch)
        iter = 0
        for data,targets in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 2)
            print("Done!")
            iter+=1
    torch.save(mozafari.state_dict(), "saved_l2.net")

# initial adaptive learning rates
apr = mozafari.stdp3.learning_rate[0][0].item()
anr = mozafari.stdp3.learning_rate[0][1].item()
app = mozafari.anti_stdp3.learning_rate[0][1].item()
anp = mozafari.anti_stdp3.learning_rate[0][0].item()

adaptive_min = 0
adaptive_int = 1
apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

# Training The Third Layer
print("Training the third layer")
if not train and os.path.isfile("saved.net"):
    print("Loading the saved network")
    mozafari.load_state_dict(torch.load("saved.net"))
else:
    try:
        for epoch in range(680):
            print("Epoch #:", epoch)
            perf_train = np.array([0.0,0.0,0.0])
            total_correct = 0
            total_loss = 0
            total_samples = 0
            for data,targets in MNIST_loader:
                perf_train_batch = train_rl(mozafari, data, targets)
                print(perf_train_batch)
                #update adaptive learning rates
                apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
                anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
                app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
                anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
                mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
                print(perf_train_batch)
                perf_train += perf_train_batch


                total_correct += perf_train_batch[0]
                total_loss += perf_train_batch[1]
                total_samples += len(data)

            perf_train /= len(MNIST_loader)
            if best_train[0] <= perf_train[0]:
                best_train = np.append(perf_train, epoch)
            print("Current Train:", perf_train)
            print("   Best Train:", best_train)

            # Save performance
            train_acc.append(total_correct/total_samples)
            train_loss.append(total_loss/total_samples)

            total_correct = 0
            total_loss = 0
            total_samples = 0
            for data,targets in MNIST_testLoader:
                perf_test = test(mozafari, data, targets)
                if best_test[0] <= perf_test[0]:
                    best_test = np.append(perf_test, epoch)
                    torch.save(mozafari.state_dict(), "saved.net")
                print(" Current Test:", perf_test)
                print("    Best Test:", best_test)
                total_correct += perf_test[0]
                total_loss += perf_test[1]
                total_samples += len(data)

            # Save performance
            test_acc.append(total_correct/total_samples)
            test_loss.append(total_loss/total_samples)

            # save performance in a unique csv file
            np.savetxt("performance.csv", np.array([train_acc, train_loss, test_acc, test_loss]).T, delimiter=",", header="train_acc,train_loss,test_acc,test_loss")

            # early stopping
            if epoch - best_test[3] > 10:
                break
    except KeyboardInterrupt:
        print("Training Interrupted")
        print("Best Train:", best_train)
        print("Best Test:", best_test)

print("Evaluation")

# Compute the confusion matrix
conf_matrix = np.zeros((10,10))
for data,targets in MNIST_testLoader:
    for i in range(len(data)):
        data_in = data[i]
        target_in = targets[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = mozafari(data_in, 3)
        if d != -1:
            conf_matrix[d][target_in] += 1

# Save the confusion matrix in png with numbers
print("Confusion Matrix:")
print(conf_matrix)
fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix)
for (i, j), val in np.ndenumerate(conf_matrix):
    ax.text(j, i, f"{val:.0f}", ha='center', va='center')
plt.xticks(range(10))
plt.yticks(range(10))
fig.colorbar(cax)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.savefig("confusion_matrix.png")