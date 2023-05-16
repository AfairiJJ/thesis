from __future__ import print_function

import argparse
from datetime import datetime, date

import joblib
import pandas as pd
import torch
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from torch.autograd.variable import Variable
from torch.optim import Adam

from Functions import metrics
from Functions.original.datasets.dataset import Dataset
from Functions.original.datasets.formats import data_formats, loaders

from Functions.original.methods.general.discriminator import Discriminator
from Functions.original.methods.general.generator import Generator
from Functions.original.methods.general.wgan_gp import calculate_gradient_penalty

from Functions.original.utils.categorical import load_variable_sizes_from_metadata
from Functions.original.utils.commandline import parse_int_list
from Functions.original.utils.cuda import to_cuda_if_available, to_cpu_if_available
from Functions.original.utils.initialization import load_or_initialize
from Functions.original.utils.logger import Logger
from Functions.original.utils.undo_dummy import back_from_dummies
from _1_DataPrep import prepare_gandata_for_regression
from config.config import *
from sampler import sample


def trainn(generator,
           discriminator,
           train_data,
           val_data,
           output_gen_path,
           output_disc_path,
           output_loss_path,
           batch_size=128,
           start_epoch=0,
           num_epochs=100,
           num_disc_steps=2,
           num_gen_steps=1,
           noise_size=128,
           l2_regularization=0.001,
           learning_rate=0.001,
           penalty=0.1,
           datacols = None,
           scaler = None
           ):
    dev_gen_old = 9999999999
    generator, discriminator = to_cuda_if_available(generator, discriminator)
    optim_gen = Adam(generator.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    optim_disc = Adam(discriminator.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    logger = Logger(output_loss_path, append=start_epoch > 0)

    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()

        # train
        generator.train(mode=True)
        discriminator.train(mode=True)

        disc_losses = []
        gen_losses = []

        more_batches = True
        train_data_iterator = train_data.batch_iterator(batch_size)

        while more_batches:
            # train discriminator
            for _ in range(num_disc_steps):
                # next batch
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    more_batches = False
                    break

                optim_disc.zero_grad()

                # first train the discriminator only with real data
                real_features = Variable(torch.from_numpy(batch))
                real_features = to_cuda_if_available(real_features)
                real_pred = discriminator(real_features)
                # real_loss = abs(real_pred - 0.9).mean(0).view(1)
                real_loss = - real_pred.mean(0).view(1) # 0->0 is worst, 1->-1 is best

                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_features = generator(noise, training=True)
                fake_features = fake_features.detach()  # do not propagate to the generator
                fake_pred = discriminator(fake_features)
                #fake_loss = abs(fake_pred).mean(0).view(1)
                fake_loss = fake_pred.mean(0).view(1) # 0->0 is best, 1->1 is worst

                # Backward propagation
                fake_loss.backward()
                real_loss.backward()

                # this is the magic from WGAN-GP
                gradient_penalty = calculate_gradient_penalty(discriminator, penalty, real_features, fake_features)
                gradient_penalty.backward()

                # finally update the discriminator weights
                # using two separated batches is another trick to improve GAN training
                optim_disc.step()

                disc_loss = real_loss + fake_loss
                disc_loss = to_cpu_if_available(disc_loss)
                disc_losses.append(disc_loss.data.numpy())

                del disc_loss
                del gradient_penalty
                del fake_loss
                del real_loss

            # train generator
            for _ in range(num_gen_steps):
                optim_gen.zero_grad()

                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                gen_features = generator(noise, training=True)
                fake_pred = discriminator(gen_features)
                fake_loss = - fake_pred.mean(0).view(1) # 1->-1 is best, 0->0 is worst
                fake_loss.backward()

                optim_gen.step()

                fake_loss = to_cpu_if_available(fake_loss)
                gen_losses.append(fake_loss.data.numpy())

                del fake_loss

        # log epoch metrics for current class
        logger.log(epoch_index, num_epochs, "discriminator.pt", "train_mean_loss", np.mean(disc_losses))
        logger.log(epoch_index, num_epochs, "generator.pt", "train_mean_loss", np.mean(gen_losses))

        # save models for the epoch
        if ((epoch_index) % 50 == 0):
            print('-----------------------------------------------')
            logger.flush()

            gendata = sample(
                generator,
                num_samples,
                num_features,
                batch_size=batch_size,
                noise_size=noise_size
            )

            formula1 = f"ClaimNb ~ VehPowerGLM + C(VehAgeGLM, Treatment(reference=2)) + C(DrivAgeGLM, Treatment(reference=5)) + BonusMalusGLM + VehGas + DensityGLM + C(Region, Treatment(reference='R24'))+ VehBrand + AreaGLM"

            data_val = pd.DataFrame(val_data.features, columns=datacols)
            data_val = prepare_gandata_for_regression(data_val, scaler)

            data_gen = pd.DataFrame(gendata, columns=datacols)
            data_gen = prepare_gandata_for_regression(data_gen, scaler)

            data_real = pd.DataFrame(train_data.features, columns=datacols)
            data_real = prepare_gandata_for_regression(data_real, scaler)

            glm_gen = smf.glm(formula=formula1, data=data_gen, family=sm.families.Poisson(link=sm.families.links.log()),
                           offset=np.log(data_gen['Exposure'])).fit()
            glm_real = smf.glm(formula=formula1, data=data_real, family=sm.families.Poisson(link=sm.families.links.log()),
                           offset=np.log(data_real['Exposure'])).fit()


            dev_gen = metrics.poisson_deviance(glm_gen.predict(data_val, offset=np.log(data_val['Exposure'])), data_val['ClaimNb'])
            dev_real = metrics.poisson_deviance(glm_real.predict(data_val, offset=np.log(data_val['Exposure'])), data_val['ClaimNb'])
            dev_base = metrics.poisson_deviance([data_real['ClaimNb'].mean()] * len(data_val), data_val['ClaimNb'])

            print(f'Poisson Deviance: Generated: {dev_gen}; Real: {dev_real}; Base: {dev_base}')

            if dev_gen < dev_gen_old:
                torch.save(generator.state_dict(), output_gen_path)
                torch.save(discriminator.state_dict(), output_disc_path)
            dev_gen_old = dev_gen

            del dev_base
            del glm_gen
            del dev_gen
            del glm_real

            del gen_features

    logger.close()


def trainnn(traindata, ss):
    options_parser = argparse.ArgumentParser(description="Train Gumbel generator and discriminator.")
    options = options_parser.parse_args()

    options.validation_proportion = 0.15
    options.noise_size = 128
    options.batch_size = 128
    options.start_epoch = 0
    options.num_epochs=10000
    options.l2_regularization=0.00001
    options.learning_rate=0.01
    options.generator_hidden_sizes='128, 128, 128' # '256,128'
    options.bn_decay = 0.9
    options.discriminator_hidden_sizes = '128,128,128' # "256,128",
    options.num_discriminator_steps = 2
    options.num_generator_steps = 1
    options.penalty = 0.01
    options.seed = seed
    options.metadata = metadata
    options.output_generator = output_generator
    options.output_discriminator = output_discriminator
    options.output_loss = output_loss

    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.seed)

    datacols = traindata.columns

    features = traindata.to_numpy()
    features = features.astype(np.float32)

    data = Dataset(features)
    train_data, val_data = data.split(1.0 - options.validation_proportion)

    variable_sizes = load_variable_sizes_from_metadata(options.metadata)

    generator = Generator(
        options.noise_size,
        variable_sizes,
        hidden_sizes=parse_int_list(options.generator_hidden_sizes),
        bn_decay=options.bn_decay
    )

    load_or_initialize(generator, state_dict_path=None)

    discriminator = Discriminator(
        features.shape[1],
        hidden_sizes=parse_int_list(options.discriminator_hidden_sizes),
        bn_decay=0,  # no batch normalization for the critic
        critic=True
    )

    load_or_initialize(discriminator, state_dict_path=None)

    trainn(
        generator,
        discriminator,
        train_data,
        val_data,
        options.output_generator,
        options.output_discriminator,
        options.output_loss,
        batch_size=options.batch_size,
        start_epoch=options.start_epoch,
        num_epochs=options.num_epochs,
        num_disc_steps=options.num_discriminator_steps,
        num_gen_steps=options.num_generator_steps,
        noise_size=options.noise_size,
        l2_regularization=options.l2_regularization,
        learning_rate=options.learning_rate,
        penalty=options.penalty,
        datacols=datacols,
        scaler=ss
    )


if __name__ == "__main__":
    trainnn()
