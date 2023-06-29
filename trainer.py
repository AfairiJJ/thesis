from __future__ import print_function

import pickle

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
    train1 = specific.inverse_transform(train1)
    test1 = specific.inverse_transform(test1)

    notintrain = [colname for colname in test1.columns if colname not in train1.columns ]
    if len(notintrain) > 0:
        print(f'WARNING: Some columns are not available yet: {notintrain}')
        train1[notintrain] = 0
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
    xgb_model.fit(train1.drop(labels='ClaimNb', axis=1), train1['ClaimNb'])

    preds = xgb_model.predict(test1.drop(labels='ClaimNb', axis=1))

    # Plotting stuff
    # plt_df = test1.copy(deep=True)
    # plt_df['preds'] = preds
    # plt_df = plt_df.groupby('DrivAge')['preds'].mean()
    # plt_df.plot()
    # plt.show()
    # train1['DrivAge'].hist()
    # test1['DrivAge'].hist()
    # plt.show()

    dev_mod = metrics.poisson_deviance(preds, test1['ClaimNb'])
    dev_base = metrics.poisson_deviance([train1['ClaimNb'].mean()] * len(test1), test1['ClaimNb'])
    dev_mae = mean_absolute_error(test1['ClaimNb'], preds)
    dev_rmse = mean_squared_error(test1['ClaimNb'], preds) ** 0.5

    return dev_mod, dev_base, dev_mae, dev_rmse

def train_generator(generator,
                    discriminator,
                    train_data,
                    beginning,
                    val_data,
                    output_gen_path,
                    output_disc_path,
                    output_loss_path,
                    output_rundata,
                    specific_dataprepper,
                    batch_size=int(cc.params['batch_size']),
                    start_epoch=0,
                    num_epochs=int(cc.params['num_epochs']),
                    num_disc_steps=int(cc.params['disc_epochs']),  # From article
                    num_gen_steps=int(cc.params['gen_epochs']),  # From article
                    penalty=cc.params['loss_penalty'],  # From article

                    ):
    # Init logger
    logger = Logger(output_loss_path, append=start_epoch > 0)


    # Give model a name
    genname = f'Generator_{cc.params["sim_num"]}'
    discname = f'Discriminator_{cc.params["sim_num"]}'

    # Run RF models on real data before we start anything
    data_real = pd.DataFrame(train_data.features)
    data_val = pd.DataFrame(val_data.features)

    # Report metrics on real dataset
    dev_real, dev_base, dev_mae, dev_rmse = run_regression(data_real, data_val, specific=specific_dataprepper)
    logger.log(0, num_epochs, genname, 'Real Poisson', dev_real)
    logger.log(0, num_epochs, genname, 'Base Poisson', dev_base)
    logger.log(0, num_epochs, genname, 'Real MAE', dev_mae)
    logger.log(0, num_epochs, genname, 'Real RMSE', dev_rmse)

    ######
    lowest_poisson_dev = cc.params['lowest_dev']
    if np.isnan(float(lowest_poisson_dev)):
        lowest_poisson_dev = 99999999
    else:
        lowest_poisson_dev = cc.params['lowest_dev']
    trackers = {
        'poisson_real': dev_real,
        'mae_real': dev_mae,
        'rmse_real': dev_rmse,
        'poisson': [],
        'rmse': [],
        'mae': [],
        'plotting_epochs': [],
        'passed_epochs': [],
        'disclosses': [],
        'genlosses': []
    }

    generator, discriminator = to_cuda_if_available(generator, discriminator)
    optim_gen = Adam(generator.parameters(), weight_decay=cc.params['gen_l2_regularization'], lr=cc.params['gen_learning_rate'])
    optim_disc = Adam(discriminator.parameters(), weight_decay=cc.params['disc_l2_regularization'], lr=cc.params['disc_learning_rate'])



    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()

        # Add epoch to tracker
        trackers['passed_epochs'] += [epoch_index]

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
                    if epoch_index >= int(cc.params['beginning_set_rounds']):
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

                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(len(batch), int(cc.params['z_size'])).normal_())
                noise = to_cuda_if_available(noise)
                fake_features = generator(noise, training=True)
                fake_features = fake_features.detach()  # do not propagate to the generator
                fake_pred = discriminator(fake_features)
                fake_loss = fake_pred.mean(0).view(1) # 0->0 is best, 1->1 is worst

                # this is the magic from WGAN-GP
                gradient_penalty = calculate_gradient_penalty(discriminator, penalty, real_features, fake_features)

                # Backward propagation
                real_loss.backward()
                fake_loss.backward()
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

                noise = Variable(torch.FloatTensor(len(batch), int(cc.params['z_size'])).normal_())
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


        #(self, epoch_index, num_epochs, model_name, metric_name, metric_value)
        logger.log(epoch_index, num_epochs, genname, "train_mean_loss", genloss)
        logger.log(epoch_index, num_epochs, discname, "train_mean_loss", discloss)
        trackers['genlosses'] += [genloss]
        trackers['disclosses'] += [discloss]

        if ((epoch_index) % 10 == 0) | debugcycle:
            print('-----------------------------------------------')
            logger.flush()
            trackers['plotting_epochs'] += [epoch_index]

            gendata = sample(
                generator,
                num_features=cc.metadata['num_features'],
                num_samples= 100000,
                batch_size= 100000,
                noise_size=int(cc.params['z_size'])
            )

            data_gen = pd.DataFrame(gendata)
            dev_gen, _, dev_mae, dev_rmse = run_regression(data_gen, data_val, specific=specific_dataprepper)

            # Report generated metrics
            logger.log(epoch_index, num_epochs, genname, 'Generated Poisson', dev_gen)
            logger.log(epoch_index, num_epochs, genname, 'Generated MAE', dev_mae)
            logger.log(epoch_index, num_epochs, genname, 'Generated RMSE', dev_rmse)
            trackers['poisson'] += [dev_gen]
            trackers['mae'] += [dev_mae]
            trackers['rmse'] += [dev_rmse]

            cc.setparams(modelid=cc.params['sim_num'], param='epochs_ran', value=epoch_index)
            if dev_gen < lowest_poisson_dev:
                lowest_poisson_dev = dev_gen
                cc.setparams(modelid=cc.params['sim_num'], param='lowest_dev', value=dev_gen)
                cc.setparams(modelid=cc.params['sim_num'], param='lowest_mae', value=dev_mae)
                cc.setparams(modelid=cc.params['sim_num'], param='lowest_rmse', value=dev_rmse)
                cc.setparams(modelid=cc.params['sim_num'], param='epoch_found', value=epoch_index)
                torch.save(generator.state_dict(), output_gen_path + '_inbetween')
                torch.save(discriminator.state_dict(), output_disc_path + '_inbetween')
            # if ((epoch_index % cc.show_plots_rounds) == 0):
            #     plt.plot(plotting_epochs, devs_poisson)
            #     plt.show()
            #
            #     plt.plot(epochs_passed, disclosses_general) # BLUE
            #     plt.plot(epochs_passed, genlosses_general) # ORANGE
            #     plt.show()

            torch.save(generator.state_dict(), output_gen_path)
            torch.save(discriminator.state_dict(), output_disc_path)

            with open(cc.params['output_rundata'], 'wb') as f:
                pickle.dump(obj=trackers, file=f)

            del data_gen
            del dev_gen
            del gendata
            del gen_features
            print('-----------------------------------------------')

        # Set that model started running
        cc.setparams(modelid=cc.params['sim_num'], param='model_started', value=True)

    logger.close()


def run_training(train_data, val_data, specific, beginning, myseed = int(cc.params['seed'])):
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        print('CUDA IS AVAILABLE')
        torch.cuda.manual_seed_all(myseed)

    if int(cc.params['num_samples']) < 500000:
        train_data = train_data.sample(int(cc.params['num_samples']), random_state=cc.params['seed'])
    print(f"Length of input data: {len(train_data)}")

    beginning = Dataset(beginning.to_numpy().astype(np.float32))
    train_data = Dataset(train_data.to_numpy().astype(np.float32))
    val_data = Dataset(val_data.to_numpy().astype(np.float32))

    generator = Generator(
        output_size=cc.metadata["variable_sizes"],
        noise_size=int(cc.params['z_size']),
        hidden_sizes=[int(x) for x in cc.params['generator_hidden_sizes'].split(',')],
        bn_decay=cc.params['gen_bn_decay']
    )

    print(generator)

    load_or_initialize(generator, state_dict_path=None)

    discriminator = Discriminator(
        leaky_param=cc.params['disc_leaky_param'],
        input_size=int(cc.metadata['num_features']),
        hidden_sizes=[int(x) for x in cc.params['discriminator_hidden_sizes'].split(',')],
        bn_decay=cc.params['disc_bn_decay'],  # no batch normalization for the critic
        critic=cc.params['critic']
    )

    print(discriminator)

    load_or_initialize(discriminator, state_dict_path=None)

    train_generator(
        generator = generator,
        discriminator=discriminator,
        train_data=train_data,
        beginning=beginning,
        val_data=val_data,
        output_gen_path=cc.params['output_generator'],
        output_disc_path=cc.params['output_discriminator'],
        output_loss_path=cc.params['output_loss'],
        output_rundata=cc.params['output_rundata'],
        specific_dataprepper=specific
    )


if __name__ == "__main__":
    run_training()
