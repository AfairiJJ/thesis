from __future__ import print_function

import argparse

import pandas as pd
import torch
import config.config as cc
import numpy as np

from torch.autograd.variable import Variable

from Functions.original.methods.general.generator import Generator

from Functions.original.utils.commandline import parse_int_list
from Functions.original.utils.cuda import to_cuda_if_available, to_cpu_if_available, load_without_cuda


def sample(generator, num_samples, num_features, batch_size=10000, noise_size=128):
    generator = to_cuda_if_available(generator)

    generator.train(mode=False)

    samples = np.zeros((num_samples, num_features), dtype=np.float32)

    start = 0
    while start < num_samples:
        with torch.no_grad():
            noise = Variable(torch.FloatTensor(batch_size, noise_size).normal_())
            noise = to_cuda_if_available(noise)
            batch_samples = generator(noise, training=False)
        batch_samples = to_cpu_if_available(batch_samples)
        batch_samples = batch_samples.data.numpy()

        # do not go further than the desired number of samples
        end = min(start + batch_size, num_samples)
        # limit the samples taken from the batch based on what is missing
        samples[start:end, :] = batch_samples[:min(batch_size, end - start), :]

        # move to next batch
        start = end

    return samples


# def generate_data():
#     generator = Generator(
#         cc.params['z_noise'],
#         cc.metadata["variable_sizes"],
#         hidden_sizes=cc.params['generator_hidden_sizes'],
#         bn_decay=cc.params['generator_bn_decay']
#     )
#
#     load_without_cuda(generator, cc.params['output_generator'])
#
#     data = sample(
#         generator,
#         cc.params['num_samples'],
#         cc.metadata['num_features'],
#         batch_size=500000,
#         noise_size=cc.params['z_noise']
#     )
#
#     print('Saving sample')
#     data = pd.DataFrame(data)
#     return data