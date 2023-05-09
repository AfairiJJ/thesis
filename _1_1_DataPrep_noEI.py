#%%

import warnings
warnings.filterwarnings('ignore')

from Functions import dataprep as prep
from sklearn.datasets import fetch_openml

# -*- coding: utf-8 -*-
"""
This script runs the multi GAN and allows you to step through each part
# divide y by exposure in xpxixpy
"""

import numpy as np
# import modules
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.float_format', lambda x: '%.2f' % x)

## Import created modules

#%%
freq0 = fetch_openml(data_id=41214, as_frame=True).data
freq1 = prep.feature_engineering_frequency_schelldorfer(freq0)
freq2 = prep.data_cleaning_frequency_schelldorfer(freq1)

freq2['ClaimNb'] = freq2['ClaimNb'].apply(lambda x: 4 if x > 4 else x)

train, test = prep.data_split_frequency_schelldorfer(freq2)
#%%
train.to_pickle('./data/common_dataprep/train.pickle')
test.to_pickle('./data/common_dataprep/test.pickle')
#%%

policy1 = pd.read_pickle("./data/common_dataprep/train.pickle")

cont_vars = ['VehPower',
                     'VehAge',
                     'DrivAge',
                     'Density',
                     'BonusMalus',
            'Exposure']
cats_vars = ["ClaimNb",
            'VehBrand',
            'VehGas',
            'Region',
             'Area'
            ]

policy1[cont_vars] = policy1[cont_vars].astype(float)
policy1[cats_vars] = policy1[cats_vars].astype('category')

policy2 = pd.get_dummies(policy1[cont_vars + cats_vars])

ss = preprocessing.MinMaxScaler()
policy3 = pd.DataFrame(ss.fit_transform(policy2), columns = policy2.columns)

assert len(policy3) == len(policy2), 'Lengths shouldnt differ'

# Take a sampel of the data for quickly training
pol_dat  = policy3#.sample(n = 10000, random_state = 1)

assert policy3['Exposure'].isna().sum() == 0, 'There should not be any empty exposures'

pol_dat.to_pickle("./data/gan_dataprep/train_gan.pickle")

import joblib
joblib.dump(ss, './data/gan_dataprep/scaler.pickle')

