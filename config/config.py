import json

import pandas as pd

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

metadata_path = './config/metadata.json'
paramspath = './config/ganruns.csv'

def getparams(path = paramspath):
    c1 = pd.read_csv(path, sep=';')
    c1 = c1.loc[c1['model_started'].isna()]
    c1 = c1.iloc[0]
    return c1

def setparams(modelid, param, value, path = paramspath):
    c1 = pd.read_csv(path, sep=';')
    c1.loc[c1['sim_num'] == modelid, param] = value
    c1.to_csv(path, sep=';', index=False)

def load_metadata(metadata_path=metadata_path):
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata

params = getparams()
metadata = load_metadata()