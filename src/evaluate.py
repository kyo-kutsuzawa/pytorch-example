#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from train_custom_loop import SimpleMLP, load_dataset


def evaluate():
    """Evaluation code.
    """
    import os
    import argparse

    # By using `argparse` module, you can specify parameters as command-line arguments.
    parser = argparse.ArgumentParser(description="Example of evaluation")
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", help="Enabling GPU calculation.")
    parser.add_argument('--model', type=str, default="result/nn_100.pth", help="Trained model.")
    parser.add_argument("--dataset", type=str, default="dataset/test.csv", help="Test dataset.")
    parser.add_argument("--n-units", type=int, default=64, help="Number of hidden units.")
    parser.add_argument("--out", default="result", help="Output directory.")
    args = parser.parse_args()

    # Setup a neural network
    in_dim = 3   # Input dimension
    out_dim = 3  # Output dimension
    model = SimpleMLP(in_dim, out_dim, args.n_units)
    model.load_state_dict(torch.load(args.model))  # Load a trained neural network

    # Enable GPU if specified
    # Note that GPUs are not required in the evaluation phase in many cases
    if args.use_gpu:
        model = model.to("cuda")  # Move the model to GPU memory

    # Load an test dataset
    eval_data = load_dataset(args.dataset)

    # Setup variables
    X = []       # Inputs
    Y_true = []  # Gournd-truth outputs
    Y = []       # Predicted outputs

    # Evaluation loop.
    for d in eval_data:
        # Get model outputs
        x = torch.from_numpy(d[0].reshape((1, -1)))
        y = model(x)

        # Record data
        X.append(d[0].flatten())
        Y_true.append(d[1].flatten())
        Y.append(y.detach().numpy().flatten())  # Using torch.Tensor.detach.numpy() to get a numpy array from a tensor

    # Convert the results to numpy arrays
    X = np.array(X)
    Y_true = np.array(Y_true)
    Y = np.array(Y)

    # Plot the outputs in a 3d space.
    # Blue dots are the ground-truth samples, while red dots are the predicted outputs.
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(1, 1, 1, projection="3d")
    ax1.scatter(Y_true[:, 0], Y_true[:, 1], Y_true[:, 2], color="b")
    ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="r")
    ax1.set_xlim((-2, 2))
    ax1.set_ylim((-2, 2))
    ax1.set_zlim((-2, 2))

    # Plot the outputs in a x-y space.
    # Blue dots are the ground-truth samples, while red dots are the predicted outputs.
    fig2 = plt.figure()
    ax2_1 = fig2.add_subplot(3, 1, 1)
    ax2_2 = fig2.add_subplot(3, 1, 2)
    ax2_3 = fig2.add_subplot(3, 1, 3)
    ax2_1.scatter(X[:, 0], Y_true[:, 0], color="b")
    ax2_2.scatter(X[:, 1], Y_true[:, 1], color="b")
    ax2_3.scatter(X[:, 2], Y_true[:, 2], color="b")
    ax2_1.scatter(X[:, 0], Y[:, 0], color="r")
    ax2_2.scatter(X[:, 1], Y[:, 1], color="r")
    ax2_3.scatter(X[:, 2], Y[:, 2], color="r")
    ax2_1.set_ylabel("x")
    ax2_2.set_ylabel("y")
    ax2_3.set_ylabel("z")

    # Save the figures
    os.makedirs(args.out, exist_ok=True)
    fig1.savefig(os.path.join(args.out, "3d.png"))
    fig2.savefig(os.path.join(args.out, "xy.png"))

    # Visualize the figures
    plt.show()


if __name__ == "__main__":
    with torch.no_grad():  # Specifying torch.no_grad() to disable gradient calculation
        evaluate()

