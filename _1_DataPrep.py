import warnings

from Functions.dataprep import feature_generation, cap_claimnb

warnings.filterwarnings('ignore')
from Functions import dataprep as prep
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn import preprocessing
import joblib

pd.set_option('display.float_format', lambda x: '%.2f' % x)


def prepare_alldata():
    freq0 = fetch_openml(data_id=41214, as_frame=True).data
    freq1 = prep.data_cleaning(freq0)
    freq2 = prep.feature_generation(freq1)

    train, test = prep.data_split_frequency_schelldorfer(freq2)

    # Dataprep for all datasets
    train.to_pickle('./data/common_dataprep/train.pickle')
    test.to_pickle('./data/common_dataprep/test.pickle')

    return train, test

def prepare_gandata_noei(policy1):
    # Specific data preparation for GAN datasets
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
    policy3 = pd.DataFrame(ss.fit_transform(policy2), columns=policy2.columns)

    assert policy3['Exposure'].isna().sum() == 0, 'There should not be any empty exposures'
    assert len(policy3) == len(policy2), 'Lengths shouldnt differ'

    policy3.to_pickle("./data/gan_dataprep/train_gan.pickle")
    joblib.dump(ss, './data/gan_dataprep/scaler.pickle')

    return policy3, ss

def prepare_gandata_for_regression(train):
    train = cap_claimnb(train)
    train = feature_generation(train)
    return train

if __name__ == '__main__':
    train, _ = prepare_alldata()
    prepare_gandata_noei(train)