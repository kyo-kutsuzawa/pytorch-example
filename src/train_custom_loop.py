#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


class SimpleMLP(nn.Module):
    """Simple example of a feedforward neural network.
    This model is a four-layer feedforward neural network with three layers of hidden layers (with ReLU actionation)
    and an output layer (no activation function).
    """

    def __init__(self, in_dim, out_dim, n_units):
        """Initialization function.
        in_dim: Number of input dimensions.
        out_dim: Number of output dimensions.
        n_units: Number of units (i.e., neurons) in each hidden layer.
        """
        super(SimpleMLP, self).__init__()

        # Define layers.
        self.l1 = nn.Linear(in_dim, n_units)
        self.l2 = nn.Linear(n_units, n_units)
        self.l3 = nn.Linear(n_units, n_units)
        self.l4 = nn.Linear(n_units, out_dim)

    def forward(self, x):
        """Forward calculation.
        x (torch.Tensor): input data with shape (batch, input_size).
            It has two axes: the 1st axis is the mini-batch axis (i.e., 1st data, 2nd data, ...);
            and the 2nd axis is the data axis (e.g. x, y, z, ...).
        return (torch.Tensor): output sample.
        """
        # Calculate the hidden layers
        h = F.relu(self.l1(x))  # ReLU activation is used in this model
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))

        # In many cases, the output layer should be a linear combination (i.e., no activation function)
        # because the output range is not bounded.
        y = self.l4(h)

        return y


def train():
    """Training code.
    """
    import argparse

    # By using `argparse` module, you can specify parameters as command-line arguments.
    parser = argparse.ArgumentParser(description="Example of training")
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", help="Enabling GPU calculation.")
    parser.add_argument("--dataset-train", type=str, default="dataset/train.csv", help="Training dataset.")
    parser.add_argument("--dataset-validation", type=str, default="dataset/validation.csv", help="Validation dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batchsize", type=int, default=64, help="Size of a mini-batch.")
    parser.add_argument("--n-units", type=int, default=64, help="Number of hidden units.")
    parser.add_argument("--out", default="result", help="Output directory.")
    args = parser.parse_args()

    # Setup a neural network
    in_dim = 3   # Input dimension
    out_dim = 3  # Output dimension
    model = SimpleMLP(in_dim, out_dim, args.n_units)

    # Enable GPU if specified
    if args.use_gpu:
        device = "cuda"
        model = model.to("cuda")  # Move the model to GPU memory
    else:
        device = "cpu"

    # Setup an optimizer.
    # The optimizer specifies the model to be trained and parameter updating method.
    optimizer = optim.Adam(model.parameters())  # Employ Adam, one of the gradient descent method

    # Load a training and validation dataset.
    train_loader = torch.utils.data.DataLoader(load_dataset(args.dataset_train), batch_size=args.batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(load_dataset(args.dataset_validation), batch_size=1000, shuffle=False)

    # Setup a loss function.
    # In this example, the mean squared error is used.
    criterion = nn.MSELoss()

    # Setup a log writer
    writer = SummaryWriter(log_dir=args.out)

    for epoch in range(1, args.epochs + 1):
        # One epoch of training loop
        train_loss = 0.0
        for data in train_loader:  # Load a mini-batch from the training dataset
            # Load a mini-batch
            x_train, y_train = data

            # Reset parameter gradients to zero
            optimizer.zero_grad()

            # Forward calculation
            y = model(x_train)

            # Calculate the training loss
            loss = criterion(y, y_train)

            # Update the parameters
            loss.backward()
            optimizer.step()

            # Print the result
            print("Epoch[{}] Loss: {:.5f}".format(epoch, loss), end="\r")

            # Accumulate the training loss
            train_loss += loss.item()

        # Record the training results
        avg_train_loss = train_loss / len(train_loader)
        print("Training Results   - Epoch: {:3d}  Avg loss: {:.5f}".format(epoch, avg_train_loss))
        writer.add_scalar("training/avg_loss", avg_train_loss, epoch)

        # One epoch of validation loop
        val_loss = 0.0
        for data in val_loader:  # Load a mini-batch from the validation dataset
            # Load a mini-batch
            x_train, y_train = data

            # Forward calculation without holding calculation graphs
            with torch.no_grad():
                y = model(x_train)

            # Calculate the validation loss
            loss = criterion(y, y_train)

            # Accumulate the validation loss
            val_loss += loss.item()  # Using torch.Tensor.item() to get a Python number from a tensor containing a single value

        # Record the validation results
        avg_val_loss = val_loss / len(val_loader)
        print("Validation Results - Epoch: {:3d}  Avg loss: {:.5f}".format(epoch, avg_val_loss))
        writer.add_scalar("validation/avg_loss", avg_val_loss, epoch)

        # Save the trained model in every 20 epochs
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.out, "nn_{}.pth".format(epoch)))

    writer.close()


def load_dataset(dirname):
    """Load a dataset.

    This example assume that the dataset is a csv file.
    In this example, the first three columns are inputs, and the next three columns are ground-truth outputs.
    """
    data = np.loadtxt(dirname, delimiter=",", dtype=np.float32)

    dataset = []
    for d in data:
        # Appending a pair of the input and output to the dataset
        io_pair = (d[0:3], d[3:6])  # Tuple of the input and output
        dataset.append(io_pair)

    return dataset


if __name__ == "__main__":
    train()

