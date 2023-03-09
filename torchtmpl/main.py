# coding: utf-8

# Standard imports
import logging
import sys

# External imports
import yaml
import wandb
import torch
import torch.nn as nn

# Local imports
from . import data
from . import models
from . import optim


def train(config):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config:
        wandb_config = config["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    optim_config = config["optim"]
    loss = optim.get_loss(optim_config["loss"])

    # Build the callbacks

    for e in range(optim_config["nepochs"]):
        # Train 1 epoch

        # Test

        # Update
        pass


def test(config):
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
