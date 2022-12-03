import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, accuracy_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from encoding import *


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
X = Xtemp[2]
Y = generateY(queries)
print('Done')

# Test Point D -- Ok
Xpred = generatePredX(sigmaConn, sigmaCond, predQueries)
print('Done')

scaler = MinMaxScaler(feature_range=(0,1))
x = scaler.fit_transform(X)
y = Y

X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size = 0.2, random_state = 0)


# other_params = {'learning_rate': 0.01, 'n_estimators': 8000, 'max_depth': 13, 'min_child_weight': 3, 'seed': 1234,
#                 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'objective': 'reg:squaredlogerror'}
#
#
# model = XGBRegressor(**other_params)
# model.fit(X_train, Y_train)
# ypred = model.predict(X_validation)
# print(ypred)
# print('MSLE of prediction on boston dataset:', mean_squared_log_error(Y_validation, ypred))
# print('\n')


cv_params = {'n_estimators': [6000,7000,8000, 9000, 1000]} #9000 is the best
cv_params = {'max_depth': [4,5,6,7,8,9], 'min_child_weight': [1,2,3,4,5,6]} #,4 is the best
# # cv_params = {'min_child_weight': [1,2,3,4,5,6]} #3 is the best
# # cv_params = {'gamma': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]} #0 is the best
# # cv_params = {'subsample': [0.6, 0.7, 0.9, 0.9], 'colsample_bytree': [0.6, 0.7, 0.9, 0.9]} #both 0.9 is the best
# cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]} #0.05 is the best
#
#
other_params = {'learning_rate': 0.05, 'n_estimators': 9000, 'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
                'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evalute_result = optimized_GBM.cv_results_['mean_test_score']
print('Each iteration result: {0}'.format(evalute_result))
print('The best argument values: {0}'.format(optimized_GBM.best_params_))
print('The best model score: {0}'.format(optimized_GBM.best_score_))


