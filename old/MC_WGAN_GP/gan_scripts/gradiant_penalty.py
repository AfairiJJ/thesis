"""
This script is an exact copy of the script found at
https://github.com/rcamino/multi-categorical-gans/tree/master/multi_categorical_gans/methods/general
"""

from __future__ import print_function

from builtins import len

import torch

from torch.autograd.variable import Variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def calculate_gradient_penalty(discriminator, penalty, real_data, fake_data):
    real_data = real_data.data
    fake_data = fake_data.data

    alpha = torch.rand(len(real_data), 1)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha.to(dev) * real_data.to(dev) + ((1 - alpha.to(dev)) * fake_data.to(dev))
    interpolates = Variable(interpolates, requires_grad=True)
    discriminator_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=discriminator_interpolates,
                                    inputs=interpolates,
                                    grad_outputs= torch.ones_like(discriminator_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty