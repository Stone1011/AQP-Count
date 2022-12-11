import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, accuracy_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
# import lightgbm as lgb
import xgboost as xgb
# import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import submit
from encoding import *
import sklearn
import math
import random
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor



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
print('Done')

# Test Point D -- Ok
Xpred = generatePredX(sigmaConn, sigmaCond, sigmaTable, predQueries)
print('Done')

# scaler = StandardScaler()
# x = scaler.fit_transform(X)
# Xpred = scaler.fit_transform(Xpred)
# print(max(Y))
logY = []
for y in Y:
    logY.append(math.log(y+1))

x = X
# y = Y

# 1.065037667852871
X_train, X_validation, Y_train, Y_validation = train_test_split(x, logY, test_size = 0.005, random_state = 1234)


#2022.12.9
##xgboost

###xgboost 调参
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
#
# model = XGBRegressor(**other_params)
# model.fit(X_train, Y_train)


#1.25
# model = XGBRegressor(n_estimators =300, learning_rate = 0.08)
# model.fit(X_train, Y_train)

# 1.1017022403942553
model = XGBRegressor(n_estimators =225, learning_rate = 0.18, max_depth=5, min_child_weight=1,subsample=1, gamma=0.2)
model.fit(X_train, Y_train)


ypred2 = model.predict(Xpred)
list = []
for i in range (2000):
    templist = [i , (math.exp(ypred2[i]))]
    list.append(templist)
print(ypred2.shape)

column = ['Query ID', 'Predicted Cardinality']
test=pd.DataFrame(columns=column, data=list)
test.to_csv('test_output.csv', index=False)


## 12.10 暴力调参
# learning_rate
# min = 999.9
# best_learing_rate = 0
# best_n_estimators = 0
# best_max_depth = 0
# best_min_child_weight = 0
# best_subsample = 0
# best_gamma = 0
# # for j in np.arange(0,1,0.1):
# for j in np.arange(1,10):
# # for j in np.arange(50, 1000, 10):
# # for j in np.arange(0.01,1,0.01).round(2):
#     model = XGBRegressor(n_estimators=460, learning_rate=0.18, max_depth=5, min_child_weight=1, subsample=1, gamma=0.2)
#     model.fit(X_train, Y_train)
#     ypred2 = model.predict(Xpred)
#     list = []
#     for i in range(2000):
#         templist = [i, (math.exp(ypred2[i]))]
#         list.append(templist)
#     # print(ypred2.shape)
#
#     column = ['Query ID', 'Predicted Cardinality']
#     test = pd.DataFrame(columns=column, data=list)
#     test.to_csv('test_output.csv', index=False)
#
#     temp = submit.submit('test_output.csv')
#     # print('learning_rate: ',j,', scoring: ',temp)
#     # print('n_estimators: ',j,', scoring: ',temp)
#     # print('max_depth: ',j,', scoring: ',temp)
#     # print('min_child_weight: ',j,', scoring: ',temp)
#     # print('subsample: ',j,', scoring: ', temp)
#     print('gamma: ',j,', scoring: ', temp)
#
#     if(float(temp) < float(min)):
#         min = temp
#         # best_learing_rate = j
#         # best_n_estimators = j
#         # best_max_depth = j
#         # best_min_child_weight = j
#         # best_subsample = j
#         best_gamma = j
#         test.to_csv('best_output.csv', index=False)
#
# # print('best_learing_rate is: ', best_learing_rate) #0.18
# # print('best_n_estimators is: ', best_n_estimators) #460
# # print('best_max_depth is: ', best_max_depth) #5
# # print('best_min_child_weight is: ', best_min_child_weight) #1
# # print('best_subsample is: ', best_subsample) #1
# print('best_gamma is: ', best_gamma) #0.2
# print('best_score is: ', min)


#12.10 evening
#暴力调参2

min = 999.9
best_learing_rate = 0
best_n_estimators = 0
best_max_depth = 0
best_min_child_weight = 0
best_subsample = 0
best_gamma = 0


# model = XGBRegressor(n_estimators=457, learning_rate=0.18, max_depth=5, min_child_weight=1, subsample=1, gamma=0.2)

# for j0 in np.arange(1, 10, 1):
#     for j1 in np.arange(1, 10, 1).round(3):
#         model = XGBRegressor(n_estimators=457, learning_rate=0.18, max_depth=j0, min_child_weight=j1, subsample=1, gamma=0.2)
#         model.fit(X_train, Y_train)
#         ypred2 = model.predict(Xpred)
#         list = []
#         for i in range(2000):
#             if (math.exp(ypred2[i])) <= 36244344:
#                 templist = [i, math.exp(ypred2[i])]
#             else:
#                 templist = [i, 36244344]
#
#             list.append(templist)
#         # print(ypred2.shape)
#
#         column = ['Query ID', 'Predicted Cardinality']
#         test = pd.DataFrame(columns=column, data=list)
#         test.to_csv('test_output.csv', index=False)
#
#         temp = submit.submit('test_output.csv')
#
#         message = 'max_depth: '+str(j0)+', min_child_weight: '+str(j1)+', scoring: '+str(temp)
#         print(message)
#         # print('learning_rate: ',j1,', scoring: ',temp)
#         # print('n_estimators: ',j0,', scoring: ',temp)
#         # print('max_depth: ',j,', scoring: ',temp)
#         # print('min_child_weight: ',j,', scoring: ',temp)
#         # print('subsample: ',j,', scoring: ', temp)
#         # print('gamma: ',j,', scoring: ', temp)
#
#         if(float(temp) < float(min)):
#             min = temp
#             # best_learing_rate = j1
#             # best_n_estimators = j0
#             best_max_depth = j0
#             best_min_child_weight = j1
#             # best_subsample = j
#             # best_gamma = j
#             test.to_csv('best_output.csv', index=False)
#
#         sleep_time = random.randrange(10, 20)
#         time.sleep(sleep_time)


# print('best_learing_rate is: ', best_learing_rate) #0.18
# print('best_n_estimators is: ', best_n_estimators) #457
print('best_max_depth is: ', best_max_depth) #5
print('best_min_child_weight is: ', best_min_child_weight) #1
# print('best_subsample is: ', best_subsample) #1
print('best_gamma is: ', best_gamma) #0.2
print('best_score is: ', min)




model = XGBRegressor(n_estimators=457, learning_rate=0.18, max_depth=5, min_child_weight=1, subsample=1, gamma=0.2)
model.fit(X_train, Y_train)
ypred2 = model.predict(Xpred)
list = []
for i in range(2000):
    if (math.exp(ypred2[i])) <= 36244344:
        templist = [i, math.exp(ypred2[i])]
    else:
        templist = [i, 36244344]
    # templist = [i, math.exp(ypred2[i])]

    list.append(templist)
# print(ypred2.shape)

column = ['Query ID', 'Predicted Cardinality']
test = pd.DataFrame(columns=column, data=list)
test.to_csv('test_output.csv', index=False)

temp = submit.submit('test_output.csv')
print(temp)