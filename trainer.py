from __future__ import print_function

import pandas as pd
import torch
import xgboost as xgb

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.autograd.variable import Variable
from torch.optim import Adam

from Functions import metrics
from Functions.original.datasets.dataset import Dataset
from Functions.original.methods.general.discriminator import Discriminator
from Functions.original.methods.general.generator import Generator
from Functions.original.methods.general.wgan_gp import calculate_gradient_penalty

from Functions.original.utils.categorical import load_variable_sizes_from_metadata
from Functions.original.utils.commandline import parse_int_list
from Functions.original.utils.cuda import to_cuda_if_available, to_cpu_if_available
from Functions.original.utils.initialization import load_or_initialize
from Functions.original.utils.logger import Logger
import config.config as cc
from sampler import sample

debugcycle = False
def run_regression(train, test, specific):
    train1 = train.copy(deep=True)
    test1 = test.copy(deep=True)
    train1 = specific.inverse_transform(train1, verbose=True)
    test1 = specific.inverse_transform(test1)

    notintest = [colname for colname in test1.columns if colname not in train1.columns ]
    if len(notintest) > 0:
        print('Some columns are not available yet')
        train1[notintest] = 0
        train1 = train1[test1.columns]

    xgb_model = xgb.XGBRegressor(objective='count:poisson',
                                 n_estimators=1200,
                                 max_depth=7,
                                 eta=0.025,
                                 colsample_bytree=0.8,
                                 subsample=0.9,
                                 min_child_weight=10,
                                 tree_method="hist",
                                 random_state=1)

    assert len(train1.columns) == len(test1.columns)
    print(f"ClaimNb Mean: {train1['ClaimNb'].mean()}")
    xgb_model.fit(train1.drop('ClaimNb', 1), train1['ClaimNb'])

    preds = xgb_model.predict(test1.drop('ClaimNb', 1))

    dev_mod = metrics.poisson_deviance(preds, test1['ClaimNb'])
    dev_base = metrics.poisson_deviance([train1['ClaimNb'].mean()] * len(test1), test1['ClaimNb'])
    dev_mae = mean_absolute_error(test1['ClaimNb'], preds)
    dev_rmse = mean_squared_error(test1['ClaimNb'], preds) ** 0.5

    return dev_mod, dev_base, dev_mae, dev_rmse

def trainn(generator,
           discriminator,
           train_data,
           beginning,
           val_data,
           output_gen_path,
           output_disc_path,
           output_loss_path,
           batch_size=cc.batch_size,
           start_epoch=0,
           num_epochs=cc.epochs,
           num_disc_steps=cc.disc_epochs, # From article
           num_gen_steps=cc.gen_epochs, # From article
           penalty=cc.loss_penalty, # From article
           specific = None
           ):
    # Run RF models on real data before we start anything
    data_real = pd.DataFrame(train_data.features)
    data_val = pd.DataFrame(val_data.features)

    dev_real, dev_base, dev_mae, dev_rmse = run_regression(data_real, data_val, specific=specific)
    print(f"Deviance Real: {dev_real}, Base: {dev_base}, Real MAE: {dev_mae}, Real RMSE: {dev_rmse}")

    ######
    dev_gen_old = 9999999999
    devs = []
    epochs = []
    epochs_general = []
    disclosses_general = []
    genlosses_general = []

    generator, discriminator = to_cuda_if_available(generator, discriminator)
    optim_gen = Adam(generator.parameters(), weight_decay=cc.gen_l2_regularization, lr=cc.gen_learning_rate)
    optim_disc = Adam(discriminator.parameters(), weight_decay=cc.disc_l2_regularization, lr=cc.disc_learning_rate)

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
        beginning_data_iterator = beginning.batch_iterator(batch_size)

        while more_batches:
            # train discriminator
            for _ in range(num_disc_steps):
                # next batch
                try:
                    if epoch_index >= 1000:
                        batch = next(train_data_iterator)
                    else:
                        batch = next(beginning_data_iterator)
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

                # Backward propagation
                real_loss.backward()

                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(len(batch), cc.noise_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_features = generator(noise, training=True)
                fake_features = fake_features.detach()  # do not propagate to the generator
                fake_pred = discriminator(fake_features)
                fake_loss = fake_pred.mean(0).view(1) # 0->0 is best, 1->1 is worst

                # Backward propagation
                fake_loss.backward()

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

                noise = Variable(torch.FloatTensor(len(batch), cc.noise_size).normal_())
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
        genloss = np.mean(gen_losses)
        discloss = np.mean(disc_losses)
        genlosses_general += [genloss]
        disclosses_general += [discloss]
        epochs_general += [epoch_index]

        logger.log(epoch_index, num_epochs, "generator.pt", "train_mean_loss", genloss)
        logger.log(epoch_index, num_epochs, "discriminator.pt", "train_mean_loss", discloss)



        # save models for the epoch

        if ((epoch_index) % 50 == 0) | debugcycle:
            print('-----------------------------------------------')
            logger.flush()

            gendata = sample(
                generator,
                cc.num_samples,
                cc.num_features,
                batch_size=500000,
                noise_size=cc.noise_size
            )

            data_gen = pd.DataFrame(gendata)
            dev_gen, _, dev_mae, dev_rmse = run_regression(data_gen, data_val, specific=specific)
            print(f"Deviance Generated: {dev_gen}; Real: {dev_real}; Base: {dev_base}; MAE: {dev_mae}; RMSE: {dev_rmse}")

            devs += [dev_gen]
            epochs += [epoch_index]
            plt.plot(epochs, devs)
            plt.show()

            plt.plot(epochs_general, disclosses_general) # BLUE
            plt.plot(epochs_general, genlosses_general) # ORANGE
            plt.show()

            if dev_gen < dev_gen_old:
                torch.save(generator.state_dict(), output_gen_path)
                torch.save(discriminator.state_dict(), output_disc_path)
            dev_gen_old = dev_gen

            del data_gen
            del dev_gen

            del gen_features

    logger.close()


def trainnn(train_data, val_data, specific, beginning):
    myseed = 1
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    beginning = Dataset(beginning.to_numpy().astype(np.float32))
    train_data = Dataset(train_data.to_numpy().astype(np.float32))
    val_data = Dataset(val_data.to_numpy().astype(np.float32))

    variable_sizes = load_variable_sizes_from_metadata(cc.metadata)

    generator = Generator(
        output_size=variable_sizes,
        noise_size=cc.noise_size,
        hidden_sizes=cc.hiddens_gen,
        bn_decay=cc.gen_bn_decay
    )

    print(generator)

    load_or_initialize(generator, state_dict_path=None)

    discriminator = Discriminator(
        input_size=cc.num_features,
        hidden_sizes=cc.hiddens_disc,
        bn_decay=cc.disc_bn_decay,  # no batch normalization for the critic
        critic=cc.critic
    )

    print(discriminator)

    load_or_initialize(discriminator, state_dict_path=None)

    trainn(
        generator,
        discriminator,
        train_data,
        beginning,
        val_data,
        cc.output_generator,
        cc.output_discriminator,
        cc.output_loss,
        specific=specific
    )


if __name__ == "__main__":
    trainnn()
