import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, accuracy_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from encoding import *
import sklearn
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Get the tensors X,Y and Xpred
queries = []
predQueries = []
with open('handout/train.csv') as file:
    line = file.readline()
    while line != '':
        queries.append(Query(line))
        line = file.readline()
with open('handout/test_without_label.csv') as file:
    line = file.readline()
    while line != '':
        predQueries.append(Query(line))
        line = file.readline()

Xtemp = generateX(queries)
sigmaConn = Xtemp[0]
sigmaCond = Xtemp[1]
sigmaTable= Xtemp[2]

X = Xtemp[3]
Y = generateY(queries)
print('Training data Done')
Xpred = generatePredX(sigmaConn, sigmaCond, sigmaTable, predQueries)
print('Testing data Done')

# Data Preprocessing
# scaler = StandardScaler()
# x = scaler.fit_transform(X)
# Xpred = scaler.fit_transform(Xpred)
logY = []
for y in Y:
    logY.append(math.log(y+1))

x = X
# y = Y

# Split the train test data
X_train, X_validation, Y_train, Y_validation = train_test_split(x, logY, test_size = 0.005, random_state = 1234)

##xgboost
# # cv_params = {'n_estimators': [500,600,700,800]} #800 is the best
# # cv_params = {'max_depth': [4,5,6,7,8,9], 'min_child_weight': [1,2,3,4,5,6]} #9,4 is the best
# # cv_params = {'gamma': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]} #0 is the best
# # cv_params = {'subsample': [0.6, 0.7, 0.9, 0.9], 'colsample_bytree': [0.6, 0.7, 0.9, 0.9]} #both 0.9 is the best
# cv_params = {'learning_rate': [0.2,0.25,0.3,0.5]} #0.05 is the best
# other_params = {'n_estimators':1000,'learning_rate': 0.2}
#
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(X_train, Y_train)
# evalute_result = optimized_GBM.cv_results_['mean_test_score']
# print('Each iteration result: {0}'.format(evalute_result))
# print('The best argument values: {0}'.format(optimized_GBM.best_params_))
# print('The best model score: {0}'.format(optimized_GBM.best_score_))
#
# model = XGBRegressor(**other_params)
# model.fit(X_train, Y_train)

# Model with best parameters
model = XGBRegressor(n_estimators=577, learning_rate=0.11, max_depth=5, min_child_weight=4, scale_pos_weight = 10, base_score = 50)
model.fit(X_train, Y_train)

# Predict
ypred = model.predict(X_validation)
print('MSLE of prediction on SQL dataset:', mean_squared_log_error(Y_validation, ypred))

ypred2 = model.predict(Xpred)
list = []
for i in range (2000):
    templist = [i , (math.exp(ypred2[i]))]
    list.append(templist)
print(ypred2.shape)

column = ['Query ID', 'Predicted Cardinality']
test=pd.DataFrame(columns=column, data=list)
test.to_csv('Group8.csv', index=False)

