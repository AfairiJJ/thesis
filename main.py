import pandas as pd
from sklearn.datasets import fetch_openml

from _1_DataPrep import CommonPrep, SpecificPrep
import config.config as cc
from trainer import trainnn

import inspect

print(inspect.getsource(cc))

df = fetch_openml(data_id=41214, as_frame=True).data

# GAN without expert input

# GAN with expert input
if True:
    common = CommonPrep()
    train, val, test = common.fit_transform(df)

    specific = SpecificPrep(gan_cats=cc.cats_vars_gan, xgb_cats=cc.cats_vars_xgb)
    specific = specific.fit(train)

    # Train for beginning only
    train_beginning = train.copy(deep=True).loc[train['ClaimNb'] > 0]
    train_beginning2 = train.copy(deep=True).loc[train['ClaimNb'] == 0].head(len(train_beginning))
    train_beginning = specific.transform(pd.concat([train_beginning, train_beginning2]))

    train = specific.transform(train)
    test = specific.transform(test)
    val = specific.transform(val)

    trainnn(train, val, specific, train_beginning)