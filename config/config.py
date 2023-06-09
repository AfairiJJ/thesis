import argparse
import json

import pandas as pd
from pandas.errors import EmptyDataError

original_data = './data/original_data/mtpl.pickle'

train_common = './data/common_dataprep/train.pickle'
test_common = './data/common_dataprep/test.pickle'
val_common = './data/common_dataprep/val.pickle'

train_specific = './data/specific_dataprep/train.pickle'
test_specific = './data/specific_dataprep/test.pickle'
val_specific = './data/specific_dataprep/val.pickle'

train_ganprep_noei = "./data/gan_dataprep/train_gan.pickle"
scaler_noei = './data/gan_dataprep/scaler.pickle'

train_ganinput_ei = 'data/gan_dataprep/train_ganinput_ei.pickle'
train_ganinput_noei = 'data/gan_dataprep/train_gan_noei.pickle'

paramspath = './config/ganruns.csv'

parser = argparse.ArgumentParser(description='GAN input')
parser.add_argument('--modelversion', default='3')
args = parser.parse_args()




def getparams(path = paramspath, modelversion = None):
    c1 = pd.read_csv(path, sep=';')
    if not modelversion:
        c1 = c1.loc[c1['model_started'].isna()]
        c1 = c1.iloc[0]
    else:
        c1 = c1.loc[c1['sim_num'] == int(modelversion)]
        c1 = c1.iloc[0]
    return c1

def setparams(modelid, param, value, path = paramspath):
    notloaded = True
    while notloaded:
        try:
            c1 = pd.read_csv(path, sep=';')
            c1.loc[c1['sim_num'] == modelid, param] = value
            c1.to_csv(path, sep=';', index=False)
            notloaded=False
        except EmptyDataError:
            print('Could not load data yet')

    assert len(c1.loc[c1['sim_num'] == modelid]) == 1, 'Cannot find the model or found multiple models'

params = getparams(modelversion=args.modelversion)

def load_metadata(metadata_path):
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata

metadata_ei = load_metadata('./config/metadata.json')
metadata_noei = load_metadata('./config/metadata_noei.json')

if params['has_ei']:
    matadata_noei = metadata_ei
    print('Has EI')
else:
    print('Has no EI')
