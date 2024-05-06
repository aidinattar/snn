import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import utils
from torchvision import transforms

use_cuda = True

# Define the network
class baselineSNN(nn.Module):
    def __init__(
        self,
        num_classes = 10,
        learning_rate_multiplier = 1,  
    ):
        super(baselineSNN, self).__init__()

        self.num_classes = num_classes

        #### LAYER 1 ####
        self.conv1 = snn.Convolution(
            in_channels=6,
            out_channels=30,
            kernel_size=5,
            weight_mean=0.8,
            weight_std=0.05
        )
        self.conv1_t = 15
        self.k1 = 5
        self.r1 = 3


        #### LAYER 2 ####
        self.conv2 = snn.Convolution(
            in_channels=30,
            out_channels=250,
            kernel_size=3,
            weight_mean=0.8,
            weight_std=0.05
        )
        self.conv2_t = 10
        self.k2 = 8
        self.r2 = 1


        #### LAYER 3 ####
        self.conv3 = snn.Convolution(
            in_channels=250,
            out_channels=200,
            kernel_size=5,
            weight_mean=0.8,
            weight_std=0.05
        )


        # STDP
        self.stdp1 = snn.STDP(
            conv_layer = self.conv1,
            learning_rate = (
                learning_rate_multiplier * 0.004,
                learning_rate_multiplier * -0.003
            ),
        )
        self.stdp2 = snn.STDP(
            conv_layer = self.conv2,
            learning_rate = (
                learning_rate_multiplier * 0.004,
                learning_rate_multiplier * -0.003
            ),
        )
        self.stdp3 = snn.STDP(
            conv_layer = self.conv3,
            learning_rate = (
                learning_rate_multiplier * 0.004,
                learning_rate_multiplier * -0.003
            ),
            use_stabilizer = False,
            lower_bound = 0.2,
            upper_bound = 0.8,
        )

        # ANTI STDP
        self.anti_stdp3 = snn.STDP(
            conv_layer = self.conv3,
            learning_rate = (
                learning_rate_multiplier * -0.004,
                learning_rate_multiplier * 0.0005
            ),
            use_stabilizer = False,
            lower_bound = 0.2,
            upper_bound = 0.8,
        )

        # adaptive learning rate
        self.max_ap = Parameter(torch.tensor([0.15]))

        # Decision map
        self.decision_map = self.generate_decision_map()

        # context parameters
        self.ctx = {
            'input_spikes': None,
            'potentials': None,
            'output_spikes': None,
            'winners': None,
        }
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0


    def generate_decision_map(self):
        """
        Generate a decision map for the network

        Returns
        -------
        list
            Decision map
        """
        decision_map = []
        neurons_per_class = 20
        for i in range(self.num_classes):
            decision_map.extend([i] * neurons_per_class)
        return decision_map
    

    def forward(
        self,
        input,
        max_layer,    
    ):
        """
        Forward pass of the network

        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        max_layer : int
            Maximum layer to be processed
        
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # padding to avoid edge effects
        input = sf.pad(
            input = input.float(),
            pad = (2, 2, 2, 2),
            value = 0
        )

        if self.training:
            # Layer 1
            # potential and spikes
            pot = self.conv1(input)
            spk, pot = sf.fire(
                potentials = pot,
                threshold = self.conv1_t,
                return_thresholded_potentials = True,
            )
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(
                        self.stdp1.learning_rate[0][0].item(),
                        device = self.stdp1.learning_rate[0][0].device
                    ) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -.75
                    self.stdp1.update_all_learning_rate(
                        ap.item(),
                        an.item()
                    )

                # inhibition
                pot = sf.pointwise_inhibition(
                    thresholded_potentials = pot
                )
                spk = pot.sign()
                winners = sf.get_k_winners(
                    potentials = pot,
                    kwta = self.k1,
                    inhibition_radius = self.r1,
                    spikes = spk
                )
                self.ctx.update({
                    "input_spikes": input,
                    "potentials": pot,
                    "output_spikes": spk,
                    "winners": winners
                })
                return spk, pot
            
            # Layer 2
            # potential and spikes
            spk_in = sf.pad(
                sf.pooling(
                    input = spk,
                    kernel_size = 2,
                    stride = 2,
                ),
                pad = (1, 1, 1, 1),
            )
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(
                potentials = pot,
                threshold = self.conv2_t,
                return_thresholded_potentials = True,
            )
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(
                        self.stdp2.learning_rate[0][0].item(),
                        device = self.stdp2.learning_rate[0][0].device
                    ) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -.75
                    self.stdp2.update_all_learning_rate(
                        ap.item(),
                        an.item()
                    )

                # inhibition
                pot = sf.pointwise_inhibition(
                    thresholded_potentials = pot
                )
                spk = pot.sign()
                winners = sf.get_k_winners(
                    potentials = pot,
                    kwta = self.k2,
                    inhibition_radius = self.r2,
                    spikes = spk
                )
                self.ctx.update({
                    "input_spikes": spk_in,
                    "potentials": pot,
                    "output_spikes": spk,
                    "winners": winners
                })
                return spk, pot
            
            # Layer 3
            spk_in = sf.pad(
                sf.pooling(
                    input = spk,
                    kernel_size = 3,
                    stride = 3,
                ),
                pad = (2, 2, 2, 2),
            )
            pot = self.conv3(spk_in)
            spk = sf.fire(
                potentials = pot,
            )
            winners = sf.get_k_winners(
                potentials = pot,
                kwta = 1,
                inhibition_radius = 0,
                spikes = spk
            )
            self.ctx.update({
                "input_spikes": spk_in,
                "potentials": pot,
                "output_spikes": spk,
                "winners": winners
            })
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
            
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(
                potentials = pot,
                threshold = self.conv1_t,
                return_thresholded_potentials = True,
            )
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(
                sf.pad(
                    sf.pooling(
                        input = spk,
                        kernel_size = 2,
                        stride = 2,
                    ),
                    pad = (1, 1, 1, 1),
                )
            )
            spk, pot = sf.fire(
                potentials = pot,
                threshold = self.conv2_t,
                return_thresholded_potentials = True,
            )
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(
                sf.pad(
                    sf.pooling(
                        input = spk,
                        kernel_size = 3,
                        stride = 3,
                    ),
                    pad = (2, 2, 2, 2),
                )
            )
            spk = sf.fire(
                potentials = pot,
            )
            winners = sf.get_k_winners(
                potentials = pot,
                kwta = 1,
                inhibition_radius = 0,
                spikes = spk
            )
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        
    def stdp(
        self,
        layer_idx,
    ):
        if layer_idx == 1:
            self.stdp1(
                self.ctx["input_spikes"],
                self.ctx["potentials"],
                self.ctx["output_spikes"],
                self.ctx["winners"],
            )
        elif layer_idx == 2:
            self.stdp2(
                self.ctx["input_spikes"],
                self.ctx["potentials"],
                self.ctx["output_spikes"],
                self.ctx["winners"],
            )
        # elif layer_idx == 3:
        #     self.stdp3.update()
        #     self.anti_stdp3.update()
        else:
            raise ValueError("Invalid layer index")
        
    def update_learning_rates(
        self,
        stdp_ap,
        stdp_an,
        anti_stdp_ap,
        anti_stdp_an,
    ):
        self.stdp3.update_all_learning_rate(
            stdp_ap,
            stdp_an
        )
        self.anti_stdp3.update_all_learning_rate(
            anti_stdp_an,
            anti_stdp_ap
        )

    def reward(self):
        self.stdp3(
            self.ctx["input_spikes"],
            self.ctx["potentials"],
            self.ctx["output_spikes"],
            self.ctx["winners"],
        )

    def punish(self):
        self.anti_stdp3(
            self.ctx["input_spikes"],
            self.ctx["potentials"],
            self.ctx["output_spikes"],
            self.ctx["winners"],
        )


def train_unsupervise(
    network,
    data,
    layer_idx,
):
    """
    Train the network using unsupervised learning

    Parameters
    ----------
    network : baselineSNN
        Network to be trained
    data : torch.Tensor
        Input data
    layer_idx : int
        Layer index

    Returns
    -------
    None
    """
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(
            data_in,
            layer_idx,
        )
        network.stdp(layer_idx)


def train_rl(
    network,
    data,
    target,
):
    """
    Train the network using reinforcement learning
    
    Parameters
    ----------
    network : baselineSNN
        Network to be trained
    data : torch.Tensor
        Input data
    target : int
        Target class

    Returns
    -------
    None
    """
    network.train()
    perf = np.array([0, 0, 0]) # [correct, wrong, silence]
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(
            data_in,
            3,
        )

        if d != -1:
            if d == target_in:
                perf[0] += 1
                network.reward()
            else:
                perf[1] += 1
                network.punish()
        else:
            perf[2] += 1
    return perf/len(data)


def test(
    network,
    data,
    target,
):
    """
    Test the network

    Parameters
    ----------
    network : baselineSNN
        Network to be tested
    data : torch.Tensor
        Input data
    target : torch.Tensor
        Target class

    Returns
    -------
    numpy.ndarray
        Performance
    """
    network.eval()
    perf = np.array([0, 0, 0]) # [correct, wrong, silence]
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(
            data_in,
            3,
        )
        if d != -1:
            if d == target_in:
                perf[0] += 1
            else:
                perf[1] += 1
        else:
            perf[2] += 1
    return perf/len(data)


class S1C1Transform:
    def __init__(\
        self,
        filter,
        timesteps = 15,
    ):
        self.grayscale = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0

    def __call__(self, img):
        # if self.cnt % 1000 == 0:
        #     print(self.cnt)
        self.cnt+=1
        img = self.grayscale(img)
        img = self.to_tensor(img) * 255
        img.unsqueeze_(0)
        img = self.filter(img)
        img = sf.local_normalization(img, 8)
        temporal_img = self.temporal_transform(img)
        return temporal_img.sign().byte()