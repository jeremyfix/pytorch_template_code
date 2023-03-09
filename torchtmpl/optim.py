# coding: utf-8

# External imports
import torch.nn as nn


def get_loss(lossname):
    return eval(f"nn.{lossname}()")
