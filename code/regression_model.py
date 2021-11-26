import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy
from collections import defaultdict
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from xgboost.sklearn import XGBRegressor

def model_builder(model_name, seed=1234):
    if model_name == 'Linear':
        model = LinearRegression()
    elif model_name == 'DecisionTree':
        model = DecisionTreeRegressor()
    elif model_name == 'SVM':
        model = SVR(C=1.0, epsilon=0.2,gamma='auto')
    elif model_name == 'KNN':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(max_depth=2, random_state=seed)
    elif model_name == 'NN':
        model = MLPRegressor( max_iter=500, random_state=seed)
    elif model_name == 'SGD':
        model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=seed)
    elif model_name == 'Ada':
        model = AdaBoostRegressor(n_estimators=100, random_state=seed)
    elif model_name == 'XGB':
        model = XGBRegressor(n_estimators=100, random_state=seed)
    else:
        raise ValueError('Wrong model name.')
    return model

def cal_loss(model, train_X, train_Y, test_X, test_Y):
    model.fit(train_X, train_Y)
    pred_Y = model.predict(test_X)
    return mean_squared_error(test_Y, pred_Y)

