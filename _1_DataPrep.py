import warnings

from Functions.dataprep import feature_generation, cap_claimnb
from Functions.original.utils.cuda import to_cpu_if_available
from Functions.original.utils.undo_dummy import back_from_dummies
from config.config import *

warnings.filterwarnings('ignore')
from Functions import dataprep as prep
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.float_format', lambda x: '%.2f' % x)

import warnings

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

import pandas as pd

def prepare_alldata():
    freq0 = fetch_openml(data_id=41214, as_frame=True).data
    freq1 = prep.data_cleaning(freq0)
    freq2 = prep.feature_generation(freq1)

    train, test = prep.data_split_frequency_schelldorfer(freq2)

    return train, test

def prepare_gandata(df1, cont_vars):
    # Specific data preparation for GAN datasets

    cats_vars = ["ClaimNb",
                 'VehBrand',
                 'VehGas',
                 'Region',
                 'Area'
                 ]

    df1[cont_vars] = df1[cont_vars].astype(float)
    df1[cats_vars] = df1[cats_vars].astype('category')

    df2 = pd.get_dummies(df1[cont_vars + cats_vars])

    ss = preprocessing.MinMaxScaler()
    df3 = pd.DataFrame(ss.fit_transform(df2), columns=df2.columns)

    assert df3['Exposure'].isna().sum() == 0, 'There should not be any empty exposures'
    assert len(df3) == len(df2), 'Lengths shouldnt differ'

    return df3, ss

def prepare_gandata_for_regression(df, ss):
    dropcols = ['EI', 'GDV']
    df = pd.DataFrame(ss.inverse_transform(df), columns=df.columns)
    df = back_from_dummies(df)
    df = df.drop(dropcols, axis='columns', errors='raise')
    df['ClaimNb'] = df['ClaimNb'].astype('float').astype('int')
    df = cap_claimnb(df)
    df = to_cpu_if_available(df)
    df = feature_generation(df)
    # X = pd.get_dummies(X, drop_first=True)

    return df

def gan_to_regression(df):


    return X, y

def add_expertinput(train, test):
    # Expert input from Isabella
    def build_predict(train, test, formula):
        glm1 = smf.glm(formula=formula, data=train, family=sm.families.Poisson(link=sm.families.links.log()),
                       offset=np.log(train['Exposure'])).fit()
        preds_train = glm1.predict(train, offset=np.log(train['Exposure']))
        preds_test = glm1.predict(test, offset=np.log(test['Exposure']))

        return preds_train, preds_test

    train['EI_Density'], test['EI_Density'] = build_predict(train, test, 'ClaimNb ~ Density')  # GLM1, do not adjust

    train['EI_DrivAge'], test['EI_DrivAge'] = build_predict(train, test,
                                                            'ClaimNb ~ DrivAge + I(DrivAge**2) + I(DrivAge**3) + I(DrivAge**4) + I(DrivAge**5)')  # GLM5, do not adjust

    train['EI_BonusMalus1'], test['EI_BonusMalus1'] = build_predict(train, test,
                                                                    'ClaimNb ~ BonusMalus + I(BonusMalus**2)')

    bm_below_100 = train.loc[train['BonusMalus'] <= 100, 'BonusMalus'].mean()
    bm_above_100 = train.loc[train['BonusMalus'] > 100, 'BonusMalus'].mean()
    train.loc[train['BonusMalus'] <= 100, 'EI_BonusMalus2'] = bm_below_100
    test.loc[test['BonusMalus'] <= 100, 'EI_BonusMalus2'] = bm_below_100
    train.loc[train['BonusMalus'] > 100, 'EI_BonusMalus2'] = bm_above_100
    test.loc[test['BonusMalus'] > 100, 'EI_BonusMalus2'] = bm_above_100

    train['EI_VehAge'], test['EI_VehAge'] = build_predict(train, test, 'ClaimNb ~ VehAge + I(VehAge**2) + I(VehAge**3)')

    train.loc[train['VehAge'] <= 5, 'EI_VehAge'] = 0.05
    test.loc[test['VehAge'] <= 5, 'EI_VehAge'] = 0.05

    # %%
    train.head()
    # %%
    train['ClaimNb'].describe()
    # %%
    # Expert Input from GDV

    # Vehicle Power
    for df in train, test:
        df['GDV_Area'] = df['Area'].copy(deep=True)
        df['GDV_Area'] = df['GDV_Area'].replace('A', 38.5)
        df['GDV_Area'] = df['GDV_Area'].replace('B', 41.5)
        df['GDV_Area'] = df['GDV_Area'].replace('C', 43.5)
        df['GDV_Area'] = df['GDV_Area'].replace('D', 46.5)
        df['GDV_Area'] = df['GDV_Area'].replace('E', (49 + 55) / 2)
        df['GDV_Area'] = df['GDV_Area'].replace('F', (58 + 65) / 2)

        # Vehicle Age
        df.loc[df['VehAge'] < 3, 'GDV_VehAge'] = 40
        df.loc[df['VehAge'] == 3, 'GDV_VehAge'] = 44
        df.loc[df['VehAge'].between(4, 5), 'GDV_VehAge'] = 46
        df.loc[df['VehAge'].between(6, 7), 'GDV_VehAge'] = 49
        df.loc[df['VehAge'] == 8, 'GDV_VehAge'] = 51
        df.loc[df['VehAge'] == 9, 'GDV_VehAge'] = 53
        df.loc[df['VehAge'].between(10, 11), 'GDV_VehAge'] = 57
        df.loc[df['VehAge'] == 12, 'GDV_VehAge'] = 61
        df.loc[df['VehAge'].between(13, 15), 'GDV_VehAge'] = 66
        df.loc[df['VehAge'].between(16, 17), 'GDV_VehAge'] = 69
        df.loc[df['VehAge'].between(18, 22), 'GDV_VehAge'] = 69
        df.loc[df['VehAge'] >= 23, 'GDV_VehAge'] = 34

        # Driver Age
        df.loc[df['DrivAge'] <= 18, 'GDV_DrivAge'] = 97
        df.loc[df['DrivAge'] == 19, 'GDV_DrivAge'] = 84
        df.loc[df['DrivAge'] == 20, 'GDV_DrivAge'] = 75
        df.loc[df['DrivAge'].between(21, 22), 'GDV_DrivAge'] = 67
        df.loc[df['DrivAge'].between(23, 24), 'GDV_DrivAge'] = 59
        df.loc[df['DrivAge'].between(25, 26), 'GDV_DrivAge'] = 67
        df.loc[df['DrivAge'].between(27, 41), 'GDV_DrivAge'] = 47
        df.loc[df['DrivAge'].between(42, 62), 'GDV_DrivAge'] = 37
        df.loc[df['DrivAge'].between(63, 67), 'GDV_DrivAge'] = 37
        df.loc[df['DrivAge'].between(68, 70), 'GDV_DrivAge'] = 40
        df.loc[df['DrivAge'].between(71, 72), 'GDV_DrivAge'] = 44
        df.loc[df['DrivAge'].between(73, 74), 'GDV_DrivAge'] = 48
        df.loc[df['DrivAge'].between(75, 76), 'GDV_DrivAge'] = 54
        df.loc[df['DrivAge'].between(77, 78), 'GDV_DrivAge'] = 58
        df.loc[df['DrivAge'].between(79, 81), 'GDV_DrivAge'] = 63
        df.loc[df['DrivAge'] >= 82, 'GDV_DrivAge'] = 74

    assert 'GDV_Area' in train.columns
    assert 'GDV_DrivAge' in train.columns

    return train, test