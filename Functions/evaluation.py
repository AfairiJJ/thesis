import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Functions import metrics

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def run_xgboost(train, test):
    xgb_model = xgb.XGBRegressor(objective='count:poisson',
                                 n_estimators=1200,
                                 max_depth=7,
                                 eta=0.025,
                                 colsample_bytree=0.8,
                                 subsample=0.9,
                                 min_child_weight=10,
                                 tree_method="hist",
                                 random_state=1)

    xgb_model.fit(train.drop(labels='ClaimNb', axis=1), train['ClaimNb'])

    preds1 = xgb_model.predict(test.drop(labels='ClaimNb', axis=1))

    dev_mod1 = metrics.poisson_deviance(preds1, test['ClaimNb'])
    dev_mae1 = mean_absolute_error(test['ClaimNb'], preds1)
    dev_rmse1 = mean_squared_error(test['ClaimNb'], preds1) ** 0.5
    gini1 = gini(preds1)

    return round(dev_mod1, 2), round(dev_mae1, 2), round(dev_rmse1, 2), round(gini1, 2)
