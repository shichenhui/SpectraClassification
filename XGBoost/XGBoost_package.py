# Using xgboost package to classisy. 
# The package introduction is in,
# https://xgboost.readthedocs.io/en/latest/python/python_api.html.


import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt
import sys
from sklearn.preprocessing import normalize
import itertools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import MyUtils


X_train, y_train = MyUtils.load_data(r'C:\Users\panda\Desktop\光谱数据样例\star_AFGK_2kx4.csv', class_num=4,
                                                     norm=True, shuffle=True, split=0)


model = xgb.XGBClassifier(
    n_jobs=30,
    learning_rate=0.1,
    n_estimators=100,
    min_child_weight=5,
    max_delta_step=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    scale_pos_weight=0.8,
    silent=True,
    missing=None,
    objective='multi:softmax',
    #eval_metric='auc',
    gamma=0,

    reg_lambda=0.4,
    max_depth = 15,
)
paras = {

}



# model.fit(X_train, y_train, eval_metric="mlogloss",early_stopping_rounds=10,
#         eval_set=[(X_test, y_test)])
# y_pred = model.predict(X_test)
#
# score = accuracy_score(y_pred, y_test)
# print(score)