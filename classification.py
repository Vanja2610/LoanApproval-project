import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('processed_data', index_col='LoanNr_ChkDgt')


class Hyperparameter_tuning:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def xgbc(self):

        grid_param = {'max_depth': [5, 6, 7]}  # this is the end point
        grid = GridSearchCV(XGBClassifier(), param_grid=grid_param, verbose=3, scoring='f1')
        # notw gridsearch also does k fold, but on xtrain, ytrain, and not X and y
        grid.fit(self.x_train, self.y_train)
        print(grid.best_params_)
        print(grid.best_score_)

    def rflc(self):

        grid_param = {'max_depth': [16,18], 'max_features': ['log2','sqrt']}  # this is the end point
        grid = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid=grid_param, verbose=3, scoring='f1')
        grid.fit(self.x_train, self.y_train)
        print(grid.best_params_)
        print(grid.best_score_)


def extreme_gradient_boost(x_train, y_train):

    xgb = XGBClassifier(max_depth=7, n_estimators=200, subsample=0.5)
    xgb.fit(x_train, y_train)
    return xgb


def random_forest(x_train, y_train):

    rf = RandomForestClassifier(max_depth=18, max_features='sqrt', n_jobs=-1) # improve speed
    rf.fit(x_train, y_train)
    return rf  # grid.cv_results_


def valuation(algorithm, X, y):  # comparing models
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(X):
        x_train, y_train = X.iloc[train_index], y.iloc[train_index]
        x_test, y_test = X.iloc[test_index], y.iloc[test_index]
        model = algorithm(x_train, y_train)
        train_f1 = f1_score(model.predict(x_train), y_train)
        test_f1 = f1_score(model.predict(x_test), y_test)
    print('train f1: ', train_f1,'test f1: ', test_f1)


X = df.drop(columns='MIS_Status')
y = df['MIS_Status']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
Tuning parameters for xgb and rf:

# Hyperparameter_tuning(x_train,y_train).xgbc()

# Hyperparameter_tuning(x_train,y_train).xgbc()

print('Valuation for xgbc: ')
valuation(extreme_gradient_boost, X, y)
final : train f1 96.3%, test f1=96.9 with default


print('Valuation for rfc: ')
valuation(random_forest, X, y)
0.9901054620137438 test_f1:  0.96187082124385
95.79,96.5
final 96.8, 96.7


very satisfied, not rly need to change parameters, decided on f1 to look as data is skewed
more 1ns than zeros
xgb is no doubt faster, however 2.5x times rf cpu used
"""

model = extreme_gradient_boost(x_train.values, y_train.values)

with open(r"finalized_model.pickle", "wb") as output_file:
    pickle.dump(model, output_file)

# maybe do clustering, mx 3

