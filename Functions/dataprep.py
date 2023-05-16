# Import necessary libraries
from copy import deepcopy

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config.config import *


def data_cleaning(freq):
    df_freq = freq.iloc[freq.drop(['IDpol', 'Exposure', 'ClaimNb'], axis=1).drop_duplicates().index]
    df_freq = df_freq.reset_index(drop=True)
    df_freq['GroupID'] = df_freq.index + 1
    df_freq = pd.merge(freq, df_freq, how='left')
    df_freq['GroupID'] = df_freq['GroupID'].fillna(method='ffill')
    
    return df_freq

def feature_generation(df_freq):
    for colname in ['ClaimNb', 'VehAge', 'DrivAge', 'VehPower']:
        df_freq[colname] = df_freq[colname].astype(int)
    df_freq = cap_claimnb(df_freq)
    df_freq['VehAge'] = df_freq['VehAge'].apply(lambda x: 20 if x > 20 else x)
    df_freq['DrivAge'] = df_freq['DrivAge'].apply(lambda x: 90 if x > 90 else x)
    df_freq['Exposure'] = df_freq['Exposure'].apply(lambda x: 1. if x > 1 else x)
    if 'Density' in df_freq.columns:
        df_freq['DensityGLM'] = df_freq['Density'].apply(lambda x: round(math.log(x), 2)) # Changed, output was saved directly to Density, now output is saved to DensityGLM
    df_freq['BonusMalusGLM'] = df_freq['BonusMalus'].apply(lambda x: 150 if x > 150 else int(x))
    df_freq['AreaGLM'] = df_freq['Area'].apply(lambda x: ord(x) - 64)
    df_freq['VehPowerGLM'] = df_freq['VehPower'].apply(lambda x: 9 if x > 9 else x).astype(int)
    # df_freq['VehPowerGLM'] = df_freq['VehPowerGLM'].apply(lambda x: str(x))
    df_freq['VehAgeGLM'] = pd.cut(df_freq['VehAge'], bins=[0, 1, 10, np.inf], labels=[1, 2, 3], include_lowest=True)
    df_freq['DrivAgeGLM'] = pd.cut(df_freq['DrivAge'], bins=[18, 21, 26, 31, 41, 51, 71, np.inf],
                                   labels=[1, 2, 3, 4, 5, 6, 7], include_lowest=True)
    return df_freq

def data_split_frequency_schelldorfer(df_freq):
    df_freq_glm = deepcopy(df_freq)
    splitter = GroupShuffleSplit(test_size=testsize, n_splits=1, random_state=seed)
    split = splitter.split(df_freq_glm, groups=df_freq_glm['GroupID'])
    train_inds, test_inds = next(split)
    train = df_freq_glm.iloc[train_inds]
    test = df_freq_glm.iloc[test_inds]

    return train, test


def cap_claimnb(df):
    df['ClaimNb'] = df['ClaimNb'].astype(float).astype(int)
    df['ClaimNb'] = df['ClaimNb'].apply(lambda x: 4 if x > 4 else x)
    return df
