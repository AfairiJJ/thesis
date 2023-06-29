import argparse
import json
from time import sleep

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

def setparams(modelid, param, value, path = paramspath, firstlen=len(pd.read_csv(paramspath, sep=';'))):
    notloaded = True
    while notloaded:
        try:
            c1 = pd.read_csv(path, sep=';')
        except EmptyDataError:
            print('Could not load data yet. Reloading')
            sleep(1)
            continue

        if len(c1.loc[c1['sim_num'] == modelid]) != 1 or len(c1) != firstlen:
            print('Ganruns.csv had wrong length. Reloading')
            sleep(1)
            continue

        # If everything works
        notloaded = False
        c1.loc[c1['sim_num'] == modelid, param] = value
        c1.to_csv(path, sep=';', index=False)


params = getparams(modelversion=args.modelversion)

if params['has_ei']:
    metadata_path = './config/metadata.json'
    print('Has EI')
else:
    metadata_path = './config/metadata_noei.json'
    print('Has no EI')

def load_metadata(metadata_path=metadata_path):
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata

metadata = load_metadata(metadata_path)

