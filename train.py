################################################################################
# Title:            train.py                                                   #
# Description:      This is the training script for the model                  #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.2                                                        #
# Usage:            python train.py                                            #
# Notes:            Added hooks, TensorBoard logging, and saving metrics       #
# Python version:   3.11.7                                                     #
################################################################################

import os
import torch
import utils
import argparse
import numpy as np
from tqdm import tqdm
from mozafari2018 import MozafariMNIST2018
from deep2024 import DeepSNN
from deepr2024 import DeepRSNN, DeepRSNN2
from inception2024 import InceptionSNN

# Set random seed for reproducibility
torch.manual_seed(42)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--model", type=str, default="mozafari2018", help="Model to use")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
    parser.add_argument("--epochs", type=int, nargs='+', default=[2, 4, 100], help="Number of epochs to train each layer")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for the data loader")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--tensorboard", action="store_true", default=False, help="Enable TensorBoard logging")
    args = parser.parse_args()

    # Print the command line arguments
    print("############################################")
    print("Program to train the model")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print("############################################")

    if not os.path.isdir("models"):
        os.mkdir("models")

    # Prepare the data
    train_loader, test_loader, metrics_loader, num_classes = utils.prepare_data(args.dataset, args.batch_size)

    # Initialize the model
    if args.model == "mozafari2018":
        model = MozafariMNIST2018(num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[2]
        max_layers = 3
    elif args.model == "deep2024":
        model = DeepSNN(num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "deepr2024":
        model = DeepRSNN(num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "deepr2024_2":
        model = DeepRSNN2(num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "inception2024":
        model = InceptionSNN(num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Move the model to the appropriate device
    model.to(args.device)

    # # Check if the model is on CUDA
    # if utils.is_model_on_cuda(model):
    #     print("Model is on CUDA")
    # else:
    #     print("Model is not on CUDA")

    # Register hooks for activation maps
    model.register_hooks()

    # # Log model to TensorBoard
    # if args.tensorboard:
    #     model.log_model(input_size=(6,15,28,28))

    #############################
    # Train the model           #
    #############################
    # First layer
    if os.path.isfile(f"models/{args.model}_{args.dataset}_first_layer.pth"):
        print("Loading first layer model")
        model.load_state_dict(
            torch.load(f"models/{args.model}_{args.dataset}_first_layer.pth"),
            strict = False
        )
    else:
        iterator = tqdm(range(args.epochs[0]), desc="Training First Layer")
        for epoch in iterator:
            i = 0
            for i, (data, _) in enumerate(train_loader):
                data = data.to(args.device)  # Ensure data is on the correct device
                model.train_unsupervised(data, layer_idx=1)
                iterator.set_postfix({"Iteration": i+1})
                i += 1
        torch.save(model.state_dict(), f"models/{args.model}_{args.dataset}_first_layer.pth")

    # second layer
    if os.path.isfile(f"models/{args.model}_{args.dataset}_second_layer.pth"):
        print("Loading second layer model")
        model.load_state_dict(
            torch.load(f"models/{args.model}_{args.dataset}_second_layer.pth"),
            strict = False
        )
    else:
        iterator = tqdm(range(args.epochs[1]), desc="Training Second Layer")
        for epoch in iterator:
            i = 0
            for data, _ in train_loader:
                data = data.to(args.device)  # Ensure data is on the correct device
                model.train_unsupervised(data, layer_idx=2)
                iterator.set_postfix({"Iteration": i+1})
                i += 1
        # save model
        torch.save(model.state_dict(), f"models/{args.model}_{args.dataset}_second_layer.pth")

    if args.model in ["deepr2024", "deepr2024_2"]:
        # third layer
        if os.path.isfile(f"models/{args.model}_{args.dataset}_third_layer.pth"):
            print("Loading third layer model")
            model.load_state_dict(
                torch.load(f"models/{args.model}_{args.dataset}_third_layer.pth"),
                strict = False
            )
        else:
            iterator = tqdm(range(args.epochs[2]), desc="Training Third Layer")
            for epoch in iterator:
                i = 0
                for data, _ in train_loader:
                    data = data.to(args.device)
                    model.train_unsupervised(data, layer_idx=3)
                    iterator.set_postfix({"Iteration": i+1})
                    i += 1
            # save model
            torch.save(model.state_dict(), f"models/{args.model}_{args.dataset}_third_layer.pth")

    if args.model in ["inception2024"]:
        # third layer
        if os.path.isfile(f"models/{args.model}_{args.dataset}_third_layer.pth"):
            print("Loading third layer model")
            model.load_state_dict(
                torch.load(f"models/{args.model}_{args.dataset}_third_layer.pth"),
                strict = False
            )
        else:
            for branch in range(3):
                iterator = tqdm(range(args.epochs[2]), desc="Training Third Layer, Branch: " + str(branch))
                for epoch in iterator:
                    i = 0
                    for data, _ in train_loader:
                        data = data.to(args.device)
                        model.train_unsupervised(data, layer_idx=3, branch_idx = branch)
                        iterator.set_postfix({"Iteration": i+1})
                        i += 1
            # save model
            torch.save(model.state_dict(), f"models/{args.model}_{args.dataset}_third_layer.pth")

        # fourth layer
        if os.path.isfile(f"models/{args.model}_{args.dataset}_fourth_layer.pth"):
            print("Loading fourth layer model")
            model.load_state_dict(
                torch.load(f"models/{args.model}_{args.dataset}_fourth_layer.pth"),
                strict = False
            )
        else:
            iterator = tqdm(range(args.epochs[3]), desc="Training Fourth Layer")
            for epoch in iterator:
                i = 0
                for data, _ in train_loader:
                    data = data.to(args.device)
                    model.train_unsupervised(data, layer_idx=4)
                    iterator.set_postfix({"Iteration": i+1})
                    i += 1
            # save model
            torch.save(model.state_dict(), f"models/{args.model}_{args.dataset}_fourth_layer.pth")

    # # Log initial embeddings
    # embeddings, metadata, label_img = utils.get_embeddings_metadata(model, train_loader, args.device)
    # model.writer.add_embedding(embeddings, metadata, label_img, global_step=0, tag='Embeddings')

    # Train the R-STDP layer
    # Set learning rates
    if args.model in ["deepr2024", "deepr2024_2"]:
        model.reset_learning_rates(3)

    adaptive_min, adaptive_int = (0, 1)
    if args.model in ["mozafari2018", "deepr2024", "deepr2024_2"]:
        apr3 = model.block3['stdp'].learning_rate[0][0].item()
        anr3 = model.block3['stdp'].learning_rate[0][1].item()
        app3 = model.block3['anti_stdp'].learning_rate[0][1].item()
        anp3 = model.block3['anti_stdp'].learning_rate[0][0].item()
    if args.model in ["deep2024", "deepr2024", "deepr2024_2"]:
        apr4 = model.block4['stdp'].learning_rate[0][0].item()
        anr4 = model.block4['stdp'].learning_rate[0][1].item()
        app4 = model.block4['anti_stdp'].learning_rate[0][1].item()
        anp4 = model.block4['anti_stdp'].learning_rate[0][0].item()

    # apr_adapt3 = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr3
    # anr_adapt3 = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr3
    # app_adapt3 = ((1.0 / 10) * adaptive_int + adaptive_min) * app3
    # anp_adapt3 = ((1.0 / 10) * adaptive_int + adaptive_min) * anp3

    # apr_adapt4 = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr4
    # anr_adapt4 = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr4
    # app_adapt4 = ((1.0 / 10) * adaptive_int + adaptive_min) * app4
    # anp_adapt4 = ((1.0 / 10) * adaptive_int + adaptive_min) * anp4

    # performance
    best_train = np.array([0., 0., 0., 0.]) # correct, total, loss, epoch
    best_test = np.array([0., 0., 0., 0.]) # correct, total, loss, epoch

    try:
        iterator = tqdm(range(epochs), desc="Training R STDP Layer")
        for epoch in iterator:
            model.epoch = epoch
            perf_train = np.array([0.0, 0.0, 0.0])
            total_correct = 0
            total_loss = 0
            total_samples = 0
            i = 0
            for k, (data, targets) in enumerate(train_loader):
                # if k == 0:
                #     model.log_inputs(data, epoch)
                
                perf_train_batch = model.train_rl(data, targets, layer_idx=max_layers)
                iterator.set_postfix({"Iteration": i+1, "Performance": perf_train_batch})
                
                if args.model in ["mozafari2018", "deepr2024", "deepr2024_2"]:
                    apr_adapt3 = apr3 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt3 = anr3 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt3 = app3 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt3 = anp3 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                    model.update_learning_rates(
                        stdp_ap = apr_adapt3,
                        stdp_an = anr_adapt3,
                        anti_stdp_ap = app_adapt3,
                        anti_stdp_an = anp_adapt3,
                        layer_idx = 3
                    )
                if args.model in ["deep2024", "deepr2024", "deepr2024_2"]:
                    apr_adapt4 = apr4 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt4 = anr4 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt4 = app4 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt4 = anp4 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                    model.update_learning_rates(
                        stdp_ap = apr_adapt4,
                        stdp_an = anr_adapt4,
                        anti_stdp_ap = app_adapt4,
                        anti_stdp_an = anp_adapt4,
                        layer_idx = 4
                    )
                perf_train += perf_train_batch

                total_correct += perf_train_batch[0]
                total_loss += perf_train_batch[1]
                total_samples += np.sum(perf_train_batch)
                i += 1

            perf_train /= len(train_loader)
            if best_train[0] <= perf_train[0]:
                best_train = np.append(perf_train, epoch)
            
            # Log training performance to TensorBoard
            if args.tensorboard:
                model.writer.add_scalar('Train/Accuracy', total_correct / total_samples, epoch)
                model.writer.add_scalar('Train/Loss', total_loss / total_samples, epoch)
            model.history['train_acc'].append(total_correct / total_samples)
            model.history['train_loss'].append(total_loss / total_samples)

            total_correct = 0
            total_loss = 0
            total_samples = 0
            for data, targets in test_loader:
                data = data.to(args.device)  # Ensure data is on the correct device
                targets = targets.to(args.device)  # Ensure targets are on the correct device
                perf_test = model.test(data, targets, layer_idx=max_layers)
                if best_test[0] <= perf_test[0]:
                    best_test = np.append(perf_test, epoch)
                    torch.save(model.state_dict(), f"models/{args.model}_{args.dataset}_best.pth")

                total_correct += perf_test[0]
                total_loss += perf_test[1]
                total_samples += np.sum(perf_test)

            # Log test performance to TensorBoard
            if args.tensorboard:
                model.writer.add_scalar('Test/Accuracy', total_correct / total_samples, epoch)
                model.writer.add_scalar('Test/Loss', total_loss / total_samples, epoch)
            model.history['test_acc'].append(total_correct / total_samples)
            model.history['test_loss'].append(total_loss / total_samples)

            # Log additional metrics to TensorBoard
            model.all_preds = []
            model.all_targets = []
            for data, targets in metrics_loader:
                data = data.to(args.device)
                targets = targets.to(args.device)
                model.compute_preds(data, targets, layer_idx=max_layers)
            metrics = model.metrics()
            model.log_tensorboard(metrics, epoch)

            # # Log embeddings at the end of each epoch
            # embeddings, metadata, label_img = utils.get_embeddings_metadata(model, train_loader, args.device)
            # model.writer.add_embedding(embeddings, metadata, label_img, global_step=epoch, tag='Embeddings')

            if epoch - best_test[3] > 10:
                break

    except KeyboardInterrupt:
        # model.file.close()
        print("Training Interrupted")
        print("Best Train:", best_train)
        print("Best Test:", best_test)

    print("Best Train:", best_train)
    print("Best Test:", best_test)

    # Save training history
    model.save_history(file_path=f"models/{args.model}_{args.dataset}_history.csv")

    # Plot training history
    model.plot_history(file_path=f"models/{args.model}_{args.dataset}_history.png")

    # Save activation maps
    model.save_activation_maps(file_path=f"models/{args.model}_{args.dataset}_activation_maps")

    # Close TensorBoard writer
    model.close_tensorboard()

if __name__ == "__main__":
    main()