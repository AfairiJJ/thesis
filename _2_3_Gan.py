# %%
# -*- coding: utf-8 -*-
"""
This script runs the multi GAN and allows you to step through each part
# divide y by exposure in xpxixpy
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
# import modules
import torch.optim as optim
from patsy import dmatrices
from tqdm import tqdm

pd.set_option('display.float_format', lambda x: '%.2f' % x)

import torch

import numpy as np
back_from_dummies
## Import created modules
from old.MC_WGAN_GP.gan_scripts.auto_loader import PolicyDataset
from old.MC_WGAN_GP.gan_scripts import Generator2
from old.MC_WGAN_GP.gan_scripts import Discriminator2
from old.MC_WGAN_GP.gan_scripts import calculate_gradient_penalty
from old.MC_WGAN_GP.gan_scripts import back_from_dummies

from torch.autograd.variable import Variable

def main():
    print(torch.cuda.is_available())

    print(torch.cuda.device_count())
    devid = torch.cuda.current_device()
    print(f"Devid: {devid}")
    torch.cuda.set_device(devid)
    print(torch.cuda.get_device_name(devid))

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    # %%

    # %%
    train = pd.read_pickle("./data/gan_dataprep/train_gan.pickle")
    val = pd.read_pickle("./data/gan_dataprep/val_gan.pickle")
    # %%
    train.head(2).transpose()
    # %%
    len(train)
    # %%
    pol_dat = train
    # %%
    formula = 'ClaimNb ~ VehBrand + VehGas + Region + AreaGLM + VehPower + VehAge + DrivAge + DensityGLM + BonusMalus'
    # %%
    # Wrangle train data
    td = back_from_dummies(train)
    td['ClaimNb'] = td['ClaimNb'].astype('float').astype('int')
    y_real, X_real = dmatrices(formula,
                               data=td,
                               return_type='dataframe')


    def xpxixpy(X, y):
        return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    xy = xpxixpy(X_real, y_real)
    disc_add_rows = xy.shape[0]

    # Fit a poisson Model
    poisson_mod = sm.GLM(y_real, X_real, family=sm.families.Poisson(), offset=td['Exposure']).fit()
    original_params = poisson_mod.params

    lower = poisson_mod.params - 1.96 * poisson_mod.bse
    upper = poisson_mod.params + 1.96 * poisson_mod.bse

    # Wrangle Test Data
    test2 = back_from_dummies(val)
    test2['ClaimNb'] = test2['ClaimNb'].astype('float').astype('int')
    y_test, X_test = dmatrices(formula,
                               data=test2,
                               return_type='dataframe')
    y_test_resp = np.squeeze(y_test) / np.squeeze(test2['Exposure'])

    # make predictions on test data with models trained on train data
    dev_real = np.mean(abs(poisson_mod.predict(X_test) - y_test))

    # %%
    """
    This next section contains everything that we can tune in the GAN
    """

    # Information about the size of the data
    data_size = pol_dat.shape[1]  # number of cols in pol_dat
    var_locs = [0, 1, 2, 3, 4, 5]  # tells us where the continous variables are

    # parameters
    z_size = 100  # how big is the random vector fed into the generator
    # we should only need the 55?
    batch_size = 1000
    temperature = None  # comes into play with the categorical activation see multioutput.py

    # Generator tuning
    gen_hidden_sizes = [100, 100, 100]
    gen_bn_decay = .90
    gen_l2_regularization = 0
    gen_learning_rate = 0.01
    noise_size = z_size
    output_size = [1, 1, 1, 1, 1, 1, 5, 11, 2, 22, 6]  # how many categories with in each variable

    assert sum(output_size) == data_size

    # Discriminator tuning
    disc_hidden_sizes = [data_size, data_size]
    disc_bn_decay = .90
    critic_bool = True  # if false then between 0 and 1
    mini_batch_bool = False
    disc_leaky_param = 0.2
    disc_l2_regularization = 0
    disc_learning_rate = 0.01
    penalty = 10  ## deals with gradiant penalty

    auto_data = PolicyDataset(pol_dat, var_locs)
    # auto_loader = DataLoader(auto_data, persistent_workers=True,
    #                          batch_size=batch_size,
    #                          pin_memory=True,
    #                          shuffle=True,
    #                          num_workers=7
    #                          )

    # initilize generator and discriminator
    generator = Generator2(
        noise_size=noise_size,
        output_size=output_size,
        hidden_sizes=gen_hidden_sizes,
        bn_decay=gen_bn_decay
    ).to(dev)

    discriminator = Discriminator2(
        input_size=data_size,
        hidden_sizes=disc_hidden_sizes,
        bn_decay=disc_bn_decay,  # no batch normalization for the critic
        critic=critic_bool,  # Do you want a critic
        leaky_param=disc_leaky_param,  # parameter for leakyRelu
        mini_batch=mini_batch_bool,  # Do you want any mini batch extras
        add_rows=disc_add_rows  # Number of rows to add if appending extra rows
    ).to(dev)

    generator = generator.to(dev)
    discriminator = discriminator.to(dev)

    optim_gen = optim.Adam(generator.parameters(),
                           weight_decay=gen_l2_regularization,
                           lr=gen_learning_rate
                           )

    optim_disc = optim.Adam(discriminator.parameters(),
                            weight_decay=disc_l2_regularization,
                            lr=disc_learning_rate
                            )

    epochs = 20000
    disc_epochs = 2
    gen_epochs = 1
    generator.train_generator(mode=True)
    discriminator.train_generator(mode=True)
    disc_losses = []
    gen_losses = []
    pois_metric = []
    # %%
    disc_loss = torch.tensor(9999)
    gen_loss = torch.tensor(9999)
    loop = tqdm(total=epochs, position=0, leave=False)

    generator = generator.to(dev)
    discriminator = discriminator.to(dev)

    cats, conts = auto_data.get_catcont()
    batch = torch.cat([conts, cats], 1)
    real_features = Variable(batch)
    real_features = real_features.to(dev)
    featlen = len(real_features)

    shuffled_features = real_features
    for epoch in range(epochs):
        # Shuffling of features
        rand_indx = torch.randperm(featlen)
        shuffled_features = shuffled_features[rand_indx]

        # For each batch
        for real_batch in torch.split(shuffled_features, batch_size):
            real_batch_size = len(real_batch)

            # Discriminator
            optim_disc.zero_grad()

            # train discriminator with real data
            real_pred = discriminator(real_batch)
            # the disc outputs high numbers if it thinks the data is real, we take the negative of this
            # Because we are minimizing loss.csv
            real_loss = -real_pred.mean(0).view(1)
            real_loss.backward()

            # then train the discriminator only with fake data
            noise = Variable(torch.FloatTensor(real_batch_size, z_size).normal_()).to(dev)
            fake_features = generator(noise, training=True)
            fake_features = fake_features.detach().to(dev)  # do not propagate to the generator
            fake_pred = discriminator(fake_features)
            fake_loss = fake_pred.mean(0).view(1)
            fake_loss.backward()

            # this is the magic from WGAN-GP
            gradient_penalty = calculate_gradient_penalty(discriminator.to(dev), penalty, real_batch.to(dev), fake_features.to(dev))
            gradient_penalty.backward()

            # finally update the discriminator weights
            optim_disc.step()

            disc_loss = real_loss + fake_loss + gradient_penalty
            disc_losses = disc_loss.item()
            # Delete to prevent memory leakage
            del gradient_penalty
            del fake_loss
            del real_loss
            del disc_loss
            del real_pred
            del noise
            del fake_features
            del fake_pred

            optim_gen.zero_grad()

            noise = Variable(torch.FloatTensor(real_batch_size, z_size).normal_()).to(dev)
            gen_features = generator(noise)
            gen_features = gen_features.to(dev)
            gen_pred = discriminator(gen_features)

            gen_loss = - gen_pred.mean(0).view(1)
            gen_loss.backward()

            optim_gen.step()

            gen_loss = gen_loss
            gen_losses = gen_loss.item()
            del gen_loss
            del noise
            del gen_features
            del gen_pred

        loop.set_description(f'epoch:{epoch}, disc_loss:{disc_losses}, gen_loss:{gen_losses}')
        loop.update(1)

        # analyze poisson regression parameters every 20 epochs
        if (epoch % 20 == 0):
            with torch.no_grad():
                generated_data = generator(Variable(torch.FloatTensor(pol_dat.shape[0], z_size).normal_()).to(dev), training=False)
            df1 = pd.DataFrame(generated_data.data.to('cpu').numpy())
            df1.columns = list(pol_dat)
            df2 = back_from_dummies(df1)
            df2['ClaimNb'] = df2['ClaimNb'].astype('float').astype('int')
            y_gen, X_gen = dmatrices(formula,
                                     data=df2,
                                     return_type='dataframe')

            # df2.to_csv(output_data_save_path)
            # Fit poisson Model
            try:
                poisson_mod_gen = sm.GLM(y_gen, X_gen, family=sm.families.Poisson(), offset=np.log(df2['Exposure'])).fit()
            except ValueError:
                continue
            # Calculate Errors
            dev_gen = np.mean(abs(poisson_mod_gen.predict(X_test) - y_test))

            #print(f'Real poisson prediction: {dev_real}')
            #print(f'Generative poisson prediction: {dev_gen}')
            #, real_poiss: {dev_real}, gen_poiss: {dev_gen}')

            if (epoch > 3):
                plt.subplot(311)
                plt.plot(pois_metric, label='train')
                plt.ylabel('poission Dif')
                plt.show()
                plt.clf()

            del poisson_mod_gen
            del generated_data
            del df1
            del df2
            # del gen_features

            #torch.save(generator.state_dict(), f='./saved_parameters/gen_test')
            # print(pois_df)

if __name__ == '__main__':
    main()