import pickle

import joblib
import pandas as pd
from sklearn.datasets import fetch_openml

from _1_DataPrep import CommonPrep, SpecificPrep
import config.config as cc
from trainer import run_training


def main():
    try:
        train = pd.read_pickle(cc.train_specific)
        val = pd.read_pickle(cc.val_specific)
        train_beginning= pd.read_pickle(cc.beginning_specific)
        specific_dataprepper = joblib.load(cc.output_scaler)

    except FileNotFoundError:
        df = fetch_openml(data_id=41214, as_frame=True).data
        common = CommonPrep()
        train, val, test = common.fit_transform(df)

        specific_dataprepper = SpecificPrep(gan_cats=cc.metadata_noei['cats_vars_gan'], xgb_cats=cc.metadata_noei['cats_vars_xgb'])
        specific_dataprepper = specific_dataprepper.fit(train)

        joblib.dump(specific_dataprepper, cc.output_scaler)

        # Train for beginning only
        train_beginning = train.copy(deep=True).loc[train['ClaimNb'] > 0]
        train_beginning2 = train.copy(deep=True).loc[train['ClaimNb'] == 0].head(len(train_beginning))

        # Transform all necessary data
        train_beginning = specific_dataprepper.transform(pd.concat([train_beginning, train_beginning2]))
        train = specific_dataprepper.transform(train)
        test = specific_dataprepper.transform(test)
        val = specific_dataprepper.transform(val)

        # Saving datasets again
        train.to_pickle(cc.train_specific)
        val.to_pickle(cc.val_specific)
        test.to_pickle(cc.test_specific)
        train_beginning.to_pickle(cc.beginning_specific)

    # Starting training run
    run_training(train, val, specific_dataprepper, train_beginning)

if __name__ == '__main__':
    main()