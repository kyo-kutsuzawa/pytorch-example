#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanSquaredError, Loss
from ignite.handlers import ModelCheckpoint
import numpy as np


class SimpleMLP(nn.Module):
    """Simple example of a feedforward neural network.
    This model is a four-layer feedforward neural network
    with three layers of hidden layers (with ReLU actionation)
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
        x (numpy.ndarray or cupy.ndarray or chainer.Variable): input data.
            It has two axes: the 1st axis is the mini-batch axis (i.e., 1st data, 2nd data, ...);
            and the 2nd axis is the data axis (i.e., x, y, z, ...).
        return (chainer.Variable): output sample.
        """
        # Calculate the hidden layers
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))

        # In many cases, the output layer should be a linear combination (i.e., no activation function)
        # because the output range is not bounded.
        y = self.l4(h)

        return y


def train():
    """Training code.
    Most codes are of initialization and training setup; the actual training loop is hidden by functions of chainer.
    Each training iteration is implemented in `SimpleUpdater.update_core()`.
    """
    import argparse

    # By using `argparse` module, you can specify parameters as command-line arguments.
    parser = argparse.ArgumentParser(description="Example of training")
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", help="GPU ID. Generally, setting 0 to use GPU, or -1 to use CPU.")
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
    optimizer = optim.Adam(model.parameters())  # Use Adam, one of the gradient descent method

    # Load a training dataset and validation dataset.
    train_loader = torch.utils.data.DataLoader(load_dataset(args.dataset_train), batch_size=args.batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(load_dataset(args.dataset_validation), batch_size=1000, shuffle=False)

    # Setup a loss function.
    # In this example, the mean squared error is used.
    loss = nn.MSELoss()

    # Setup a trainer.
    trainer = create_supervised_trainer(model, optimizer, loss, device)

    # Setup an evaluator.
    metrics = {
        'accuracy': MeanSquaredError(),
        'nll': Loss(loss)
    }
    evaluator = create_supervised_evaluator(model, metrics, device)

    # Setup a log writer
    writer = SummaryWriter(log_dir=args.out)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.5f}".format(trainer.state.epoch, trainer.state.output), end="\r")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results   - Epoch: {:3d}  Avg accuracy: {:.5f} Avg loss: {:.5f}".format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))
        writer.add_scalar("training/avg_loss", metrics['nll'], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {:3d}  Avg accuracy: {:.5f} Avg loss: {:.5f}".format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))
        writer.add_scalar("validation/avg_loss", metrics['accuracy'], trainer.state.epoch)

    # Settings of model saving
    handler = ModelCheckpoint(dirname=args.out, filename_prefix='sample', save_interval=10, n_saved=3, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})

    # Start training
    trainer.run(train_loader, max_epochs=args.epochs)

    writer.close()


def load_dataset(dirname):
    """Load a dataset.

    This example assume that the dataset is a csv file.
    In this example, the first three columns are inputs,
    and the next three columns are ground-truth outputs.
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
