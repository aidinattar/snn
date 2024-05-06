"""
train.py
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms
from deepSNN import deepSNN
from deepSNN import S1C1Transform
from deepSNN import train_unsupervise
from deepSNN import train_rl, train_rl_separate
from deepSNN import test
from deepSNN import custom_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--epochs_1", type=int, default=2, help="Number of epochs")
    parser.add_argument("--epochs_2", type=int, default=4, help="Number of epochs")
    parser.add_argument("--epochs_3", type=int, default=600, help="Number of epochs")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every n epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA", default=True)
    args = parser.parse_args()

    # Check if CUDA is available
    if args.use_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    
    # print arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    # Set random seed
    torch.manual_seed(args.seed)

    kernels = [
        utils.DoGKernel(3,3/9,6/9),
        utils.DoGKernel(3,6/9,3/9),
        utils.DoGKernel(7,7/9,14/9),
        utils.DoGKernel(7,14/9,7/9),
        utils.DoGKernel(13,13/9,26/9),
        utils.DoGKernel(13,26/9,13/9)
    ]
    filter = utils.Filter(
        kernels,
        padding = 6,
        thresholds = 50
    )
    s1c1 = S1C1Transform(filter)

    # Load dataset
    if args.dataset == "mnist":
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
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size = args.batch_size,
            shuffle = True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size = len(test_data),
            shuffle = False
        )
    elif args.dataset == "cifar10":
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
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size = args.batch_size,
            shuffle = True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size = len(test_data),
            shuffle = False
        )
    elif args.dataset == "emnist":
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
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size = args.batch_size,
            shuffle = True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size = len(test_data),
            shuffle = False
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    

    # Load model
    snn = deepSNN(
        num_classes = num_classes,
        learning_rate_multiplier = 1.0,
    )

    if args.use_cuda:
        snn.cuda()

    # print summary of the model
    # snn.summary()
    custom_summary(snn)

    # save accuracy and loss
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    # Train model
    # first layer
    print("Training first layer")
    if os.path.isfile(f"models/deepSNN_{args.dataset}_first_layer.pth"):
        snn.load_state_dict(torch.load(f"models/deepSNN_{args.dataset}_first_layer.pth"))
    else:
        for epoch in range(args.epochs_1):
            print(f"Epoch {epoch}")
            iteration = 0
            for data, targets in train_loader:
                print(f"Iteration {iteration}")
                train_unsupervise(
                    network = snn,
                    data = data,
                    layer_idx = 1,
                )
                iteration += 1
            if epoch % args.save_every == 0:
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(snn.state_dict(), f"models/deepSNN_{args.dataset}_first_layer.pth")
        # save model
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(snn.state_dict(), f"models/deepSNN_{args.dataset}_first_layer.pth")

    # second layer
    print("Training second layer")
    if os.path.isfile(f"models/deepSNN_{args.dataset}_second_layer.pth"):
        snn.load_state_dict(torch.load(f"models/deepSNN_{args.dataset}_second_layer.pth"))
    else:
        for epoch in range(args.epochs_2):
            print(f"Epoch {epoch}")
            iteration = 0
            for data, targets in train_loader:
                print(f"Iteration {iteration}")
                train_unsupervise(
                    network = snn,
                    data = data,
                    layer_idx = 2,
                )
                iteration += 1
            if epoch % args.save_every == 0:
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(snn.state_dict(), f"models/deepSNN_{args.dataset}_second_layer.pth")

        # save model
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(snn.state_dict(), f"models/deepSNN_{args.dataset}_second_layer.pth")

    # adaptive learning rates
    apr = snn.stdp3.learning_rate[0][0].item()
    anr = snn.stdp3.learning_rate[0][1].item()
    app = snn.anti_stdp3.learning_rate[0][1].item()
    anp = snn.anti_stdp3.learning_rate[0][0].item()

    adaptive_min = 0
    adaptive_int = 1
    apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
    anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
    app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
    anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

    # performance
    best_train = np.array([0., 0., 0., 0.]) # correct, total, loss, epoch
    best_test = np.array([0., 0., 0., 0.]) # correct, total, loss, epoch

    # third, fourth and fifth layer
    print("Training the reward based layers")
    try:
        for epoch in range(args.epochs_3):
            print(f"Epoch {epoch}")
            performance_train = np.array([0., 0., 0.]) # correct, total, loss, epoch
           
            # train
            for data, targets in train_loader:
                # perf_train_batch = train_rl(
                #     network = snn,
                #     data = data,
                #     target = targets,
                #     max_layer = 5,
                # )
                for layer_idx in range(3, 4):
                    perf_train_batch = train_rl_separate(
                        network = snn,
                        data = data,
                        target = targets,
                        max_layer = layer_idx,
                    )
                    print(f"Layer {layer_idx} Performance: {perf_train_batch}")

                # update learning rates
                apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
                anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
                app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
                anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
                snn.update_learning_rates(
                    stdp_ap = apr_adapt,
                    stdp_an = anr_adapt,
                    anti_stdp_ap = app_adapt,
                    anti_stdp_an = anp_adapt
                )
                performance_train += perf_train_batch

            performance_train /= len(train_loader)
            if best_train[0] < performance_train[0]:
                best_train = np.append(performance_train, epoch)
            print(f"Current train accuracy: {performance_train}")
            print(f"Best train accuracy: {best_train}")

            # save performance
            train_accuracy.append(performance_train[0])
            train_loss.append(performance_train[1])

            # total_correct = 0
            # total_loss = 0
            # total = 0
            for data, targets in test_loader:
                performance_test = test(
                    network = snn,
                    data = data,
                    target = targets,
                )
                if best_test[0] <= performance_test[0]:
                    best_test = np.append(performance_test, epoch)
                    if not os.path.isdir("models"):
                        os.mkdir("models")
                    torch.save(snn.state_dict(), f"models/deepSNN_{args.dataset}__best_model.pth")
                # total_correct += performance_test[0]
                # total_loss += performance_test[1]
                # total += len(data)
                print(f"Current test accuracy: {performance_test}")
                print(f"Best test accuracy: {best_test}")
            
            # save performance
            test_accuracy.append(performance_test[0])
            test_loss.append(performance_test[1])

            # save performance in a csv file
            np.savetxt(f"{args.dataset}_performance.csv", np.array([train_accuracy, train_loss, test_accuracy, test_loss]), delimiter=",")
    
            # early stopping
            if epoch - best_test[3] > 15:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("Training interrupted")
        print(f"Best train accuracy: {best_train}")
        print(f"Best test accuracy: {best_test}")

    # save model
    if not os.path.isdir("models"):
        os.mkdir("models")
    torch.save(snn.state_dict(), f"models/deepSNN_{args.dataset}_third_layer.pth")

    print("Evaluation")
    # compute the confusion matrix
    confusion_matrix = np.zeros((
        num_classes,
        num_classes
    ))
    for data, targets in test_loader:
        for i in range(len(data)):
            data_in = data[i] 
            target_in = targets[i]
            if args.use_cuda:
                data_in = data_in.cuda()
                target_in = target_in.cuda()
            output = snn(data_in, 3)
            if output != -1:
                confusion_matrix[output][target_in] += 1
    
    # save confusion matrix in png
    plt.figure()
    plt.imshow(confusion_matrix)
    plt.colorbar()
    plt.savefig(f"{args.dataset}_confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved in confusion_matrix.png")
    print("Done")