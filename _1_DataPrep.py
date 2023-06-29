import math
from copy import deepcopy

from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from Functions.original.utils.undo_dummy import back_from_dummies
import config.config as cc

import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import warnings

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')


def claimnb_as_int(col):
 return col.astype(float).round().astype(int)

class Dummifier(TransformerMixin):
    def __init__(self, convert_columns, drop_first):
        self.convert_columns = convert_columns
        self.drop_first = drop_first

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy(deep=True)
        if 'ClaimNb' in self.convert_columns:
            X['ClaimNb'] = X['ClaimNb'].astype('category')
        else:
            X['ClaimNb'] = claimnb_as_int(X['ClaimNb'])
        X = pd.get_dummies(X, columns=self.convert_columns, sparse=False, drop_first=self.drop_first)
        return X

    def inverse_transform(self, X, y=None):
        X = X.copy(deep=True)
        X = back_from_dummies(X)
        if 'ClaimNb' in X.columns:
            X['ClaimNb'] = claimnb_as_int(X['ClaimNb'])

        return X

class ExpertInputter(TransformerMixin):
    def fit(self, X, y=None):
        self.density = self.build(X, 'ClaimNb ~ Density')
        self.drivage = self.build(X, 'ClaimNb ~ DrivAge + I(DrivAge**2) + I(DrivAge**3) + I(DrivAge**4) + I(DrivAge**5)')
        self.bonusmalus1 = self.build(X, 'ClaimNb ~ BonusMalus + I(BonusMalus**2)')
        self.vehage = self.build(X, 'ClaimNb ~ VehAge + I(VehAge**2) + I(VehAge**3)')
        self.bm_above_100 = X.loc[X['BonusMalus'] > 100, 'ClaimNb'].mean()
        self.bm_below_100 = X.loc[X['BonusMalus'] <= 100, 'ClaimNb'].mean()

        return self

    def transform(self, X, y=None):
        X = X.copy(deep=True)

        # Isabella Input
        X.loc[X['BonusMalus'] <= 100, 'EI_BonusMalus2'] = self.bm_above_100
        X.loc[X['BonusMalus'] > 100, 'EI_BonusMalus2'] = self.bm_below_100
        X['EI_Density'] = self.density.predict(X)
        X['EI_DrivAge'] = self.drivage.predict(X)
        X['EI_BonusMalus1'] = self.bonusmalus1.predict(X)
        X['EI_VehAge'] = self.vehage.predict(X)
        X.loc[X['VehAge'] <= 5, 'EI_VehAge'] = 0.05

        df = X

        # # GDV input
        # df['GDV_Area'] = df['Area'].copy(deep=True)
        # df['GDV_Area'] = df['GDV_Area'].replace(1, 38.5)
        # df['GDV_Area'] = df['GDV_Area'].replace(2, 41.5)
        # df['GDV_Area'] = df['GDV_Area'].replace(3, 43.5)
        # df['GDV_Area'] = df['GDV_Area'].replace(4, 46.5)
        # df['GDV_Area'] = df['GDV_Area'].replace(5, (49 + 55) / 2)
        # df['GDV_Area'] = df['GDV_Area'].replace(6, (58 + 65) / 2)
        #
        # # Vehicle Age
        # df.loc[df['VehAge'] < 3, 'GDV_VehAge'] = 40
        # df.loc[df['VehAge'] == 3, 'GDV_VehAge'] = 44
        # df.loc[df['VehAge'].between(4, 5), 'GDV_VehAge'] = 46
        # df.loc[df['VehAge'].between(6, 7), 'GDV_VehAge'] = 49
        # df.loc[df['VehAge'] == 8, 'GDV_VehAge'] = 51
        # df.loc[df['VehAge'] == 9, 'GDV_VehAge'] = 53
        # df.loc[df['VehAge'].between(10, 11), 'GDV_VehAge'] = 57
        # df.loc[df['VehAge'] == 12, 'GDV_VehAge'] = 61
        # df.loc[df['VehAge'].between(13, 15), 'GDV_VehAge'] = 66
        # df.loc[df['VehAge'].between(16, 17), 'GDV_VehAge'] = 69
        # df.loc[df['VehAge'].between(18, 22), 'GDV_VehAge'] = 69
        # df.loc[df['VehAge'] >= 23, 'GDV_VehAge'] = 34
        #
        # # Driver Age
        # df.loc[df['DrivAge'] <= 18, 'GDV_DrivAge'] = 97
        # df.loc[df['DrivAge'] == 19, 'GDV_DrivAge'] = 84
        # df.loc[df['DrivAge'] == 20, 'GDV_DrivAge'] = 75
        # df.loc[df['DrivAge'].between(21, 22), 'GDV_DrivAge'] = 67
        # df.loc[df['DrivAge'].between(23, 24), 'GDV_DrivAge'] = 59
        # df.loc[df['DrivAge'].between(25, 26), 'GDV_DrivAge'] = 67
        # df.loc[df['DrivAge'].between(27, 41), 'GDV_DrivAge'] = 47
        # df.loc[df['DrivAge'].between(42, 62), 'GDV_DrivAge'] = 37
        # df.loc[df['DrivAge'].between(63, 67), 'GDV_DrivAge'] = 37
        # df.loc[df['DrivAge'].between(68, 70), 'GDV_DrivAge'] = 40
        # df.loc[df['DrivAge'].between(71, 72), 'GDV_DrivAge'] = 44
        # df.loc[df['DrivAge'].between(73, 74), 'GDV_DrivAge'] = 48
        # df.loc[df['DrivAge'].between(75, 76), 'GDV_DrivAge'] = 54
        # df.loc[df['DrivAge'].between(77, 78), 'GDV_DrivAge'] = 58
        # df.loc[df['DrivAge'].between(79, 81), 'GDV_DrivAge'] = 63
        # df.loc[df['DrivAge'] >= 82, 'GDV_DrivAge'] = 74

        return df

    def build(self, df, formula):
        glm = smf.glm(formula=formula, data=df, family=sm.families.Poisson(link=sm.families.links.log())).fit()
        return glm

class CommonPrep(TransformerMixin):
    def __init__(self):
        self.testsize = 0.2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy(deep=True)

        X = self.dedupe(X)
        X = self.clean(X)
        df, test = self.split(X)
        train, val = self.split(df)

        return train, val, test

    def dedupe(self, freq):
        df_freq = freq.iloc[freq.drop(['IDpol', 'Exposure', 'ClaimNb'], axis=1).drop_duplicates().index]
        df_freq = df_freq.reset_index(drop=True)
        df_freq['GroupID'] = df_freq.index + 1
        df_freq = pd.merge(freq, df_freq, how='left')
        df_freq['GroupID'] = df_freq['GroupID'].fillna(method='ffill')

        return df_freq

    def clean(self, df_freq):
        df_freq = df_freq.copy(deep=True)
        df_freq['ClaimNb'] = df_freq['ClaimNb'].apply(lambda x: 4 if x > 4 else x)
        df_freq['Area'] = df_freq['Area'].apply(lambda x: ord(x) - 64)
        df_freq['VehAge'] = df_freq['VehAge'].apply(lambda x: 20 if x > 20 else x)
        df_freq['DrivAge'] = df_freq['DrivAge'].apply(lambda x: 90 if x > 90 else x)
        df_freq['BonusMalus'] = df_freq['BonusMalus'].apply(lambda x: 150 if x > 150 else int(x))
        df_freq['Density'] = df_freq['Density'].apply(lambda x: round(math.log(x), 2))
        df_freq['Exposure'] = df_freq['Exposure'].apply(lambda x: 1. if x > 1 else x)
        df_freq['VehPower'] = df_freq['VehPower'].apply(lambda x: 9 if x > 9 else x)
        return df_freq

    def split(self, df_freq, seed = int(cc.params['seed'])):
        df_freq_glm = deepcopy(df_freq)
        splitter = GroupShuffleSplit(test_size=self.testsize, n_splits=1, random_state=seed)
        split = splitter.split(df_freq_glm, groups=df_freq_glm['GroupID'])
        train_inds, test_inds = next(split)
        train = df_freq_glm.iloc[train_inds]
        test = df_freq_glm.iloc[test_inds]

        return train, test

class MyStandardScaler(TransformerMixin):
    def __init__(self, columns_to_scale = cc.metadata['standardized_vars']):
        self.scaler = StandardScaler(copy=True)
        self.columns_to_scale = columns_to_scale
    def fit(self, X, y=None):
        self.scaler = self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X, y=None):
        X = X.copy(deep=True)

        X[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X

    def inverse_transform(self, X, y=None):
        X[self.columns_to_scale] = self.scaler.inverse_transform(X[self.columns_to_scale])
        return X

class SpecificPrep(TransformerMixin):
    def __init__(self, gan_cats, xgb_cats, addei = cc.params['has_ei'], variables = cc.metadata['variables'], columns_to_scale = cc.metadata['standardized_vars']):
        self.dummified_cols = None
        self.scaler = MyStandardScaler(columns_to_scale=columns_to_scale)
        self.gan_dummifier = Dummifier(convert_columns=gan_cats, drop_first=False)
        self.xgb_dummifier = Dummifier(convert_columns=xgb_cats, drop_first=True)
        self.addei = addei
        self.variables = variables

        if self.addei:
            self.eier = ExpertInputter()

    def fit(self, X, y=None):
        if self.addei:
            self.eier.fit(X)
        self.gan_dummifier.fit(X)
        self.xgb_dummifier.fit(X)

        if self.addei:
            X = self.eier.transform(X)
        X = X[self.variables]
        X = self.gan_dummifier.transform(X)
        self.dummified_cols = X.columns
        self.scaler.fit(X)

        return self

    def transform(self, X, y=None):
        X = X.copy(deep=True)

        if self.addei:
            X = self.eier.transform(X)
        X = X[self.variables]
        X = self.gan_dummifier.transform(X)
        X = self.scaler.transform(X)

        return X

    def inverse_transform(self, X, y=None):
        df = pd.DataFrame(X).copy(deep=True)
        df.columns = self.dummified_cols
        df = self.scaler.inverse_transform(df)
        ei = df.filter(like="EI_", axis=1)
        df = self.gan_dummifier.inverse_transform(df)
        df = df.drop(['EI', 'GDV', 'Exposure'], axis='columns', errors='ignore')
        if self.addei:
            df = pd.concat([df, ei], axis=1)
        df = self.xgb_dummifier.transform(df)

        return df







