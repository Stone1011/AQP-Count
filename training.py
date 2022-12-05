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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor


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

scaler = StandardScaler()
x = scaler.fit_transform(X)
x = X
y = Y

X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size = 0.01, random_state = 1234)

#XGBOOST
# other_params = {'learning_rate': 0.05, 'n_estimators':9000, 'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
#                 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'objective': 'reg:squaredlogerror'}
#
#
# model = XGBRegressor(**other_params)
# model.fit(X_train, Y_train)
# ypred = model.predict(X_validation)
# print(Y_validation)
# print(ypred)
# print('MSLE of prediction on boston dataset:', mean_squared_log_error(Y_validation, ypred))
# print('\n')
#
#
# scaler = MinMaxScaler(feature_range=(0,1))
# Xpred_stad = scaler.fit_transform(Xpred)
# ypred2 = model.predict(Xpred_stad)
# list = []
# for i in range (2000):
#     templist = [i ,ypred2[i]]
#     list.append(templist)
# print(ypred2.shape)
#
# column = ['Query ID', 'Predicted Cardinality']
# test=pd.DataFrame(columns=column, data=list)
# test.to_csv('test_output.csv', index=False)
#
#
# # print('MSLE of prediction on boston dataset:', mean_squared_log_error(Y_validation, ypred))
# # print('\n')
#
# # cv_params = {'n_estimators': [6000,7000,8000, 9000, 1000]} #9000 is the best
# # cv_params = {'max_depth': [4,5,6,7,8,9], 'min_child_weight': [1,2,3,4,5,6]} #9,4 is the best
# # # # cv_params = {'gamma': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]} #0 is the best
# # # # cv_params = {'subsample': [0.6, 0.7, 0.9, 0.9], 'colsample_bytree': [0.6, 0.7, 0.9, 0.9]} #both 0.9 is the best
# # # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]} #0.05 is the best
# # #
# # #
# # other_params = {'learning_rate': 0.05, 'n_estimators': 9000, 'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
# #                 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# #
# # model = XGBRegressor(**other_params)
# # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
# # optimized_GBM.fit(X_train, Y_train)
# # evalute_result = optimized_GBM.cv_results_['mean_test_score']
# # print('Each iteration result: {0}'.format(evalute_result))
# # print('The best argument values: {0}'.format(optimized_GBM.best_params_))
# # print('The best model score: {0}'.format(optimized_GBM.best_score_))


# knn
# model = KNeighborsRegressor(n_neighbors=5, weights='distance',p=2)
# # model  = RadiusNeighborsRegressor()
# #定义一个knn分类器对象
#
# model.fit(X_train, Y_train)
# #调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
#
# ypred = model.predict(X_validation)
#  #调用该对象的测试方法，主要接收一个参数：测试数据集
# # probility=model.predict_proba(X_validation)
#  #计算各测试样本基于概率的预测
# # neighborpoint=knn.kneighbors(iris_x_test[-1:],5,False)
# #计算与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组
# score=model.score(X_validation,Y_validation,sample_weight=None)
# #调用该对象的打分方法，计算出准确率
#
# print('MSLE of prediction on boston dataset:', mean_squared_log_error(Y_validation, ypred))
# print('\n')
# print('score is:', score)


#svm
#进行模型训练
#1.线性回归
# from sklearn import linear_model
# #from sklearn import model_selection
# from sklearn.linear_model import LinearRegression
#
# model = linear_model.LinearRegression()
# #进行训练
# model.fit(X_train, Y_train)
# #通过model的coef_属性获得权重向量,intercept_获得b的值
# print("权重向量:%s, b的值为:%.2f" % (model.coef_, model.intercept_))
# #计算出损失函数的值
# print("损失函数的值: %.2f" % np.mean((model.predict(X_validation) - Y_validation) ** 2))
# #计算预测性能得分
# print("预测性能得分: %.2f" % model.score(X_validation, Y_validation))


#forest
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# random_seed=44
# random_forest_seed=np.random.randint(low=1,high=230)
# model = RandomForestRegressor(n_estimators=200, min_samples_split= 2, max_features='auto', max_depth=270, criterion='mse', random_state=1, n_jobs=-1)
# model.fit(X_train, Y_train)
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_validation)

# Search optimal hyperparameter

# n_estimators_range=[int(x) for x in np.linspace(start=50,stop=3000,num=60)]
# max_features_range=['auto','sqrt']
# max_depth_range=[int(x) for x in np.linspace(10,500,num=50)]
# max_depth_range.append(None)
# min_samples_split_range=[2,5,10]
# min_samples_leaf_range=[1,2,4,8]
# bootstrap_range=[True,False]
#
# random_forest_hp_range={'n_estimators':n_estimators_range,
#                         'max_features':max_features_range,
#                         'max_depth':max_depth_range,
#                         'min_samples_split':min_samples_split_range,
#                         # 'min_samples_leaf':min_samples_leaf_range
#                         # 'bootstrap':bootstrap_range
#                         }
# print(random_forest_hp_range)
#
# random_forest_model_test_base=RandomForestRegressor()
# random_forest_model_test_random=RandomizedSearchCV(estimator=random_forest_model_test_base,
#                                                    param_distributions=random_forest_hp_range,
#                                                    n_iter=200,
#                                                    n_jobs=-1,
#                                                    cv=3,
#                                                    verbose=1,
#                                                    random_state=random_forest_seed
#                                                    )
# random_forest_model_test_random.fit(X_train,Y_train)
#
# best_hp_now=random_forest_model_test_random.best_params_
# print(best_hp_now)
#{'n_estimators': 200, 'min_samples_split': 2, 'max_features': 'auto', 'max_depth': 270}



# MLP network
# not as good as forest
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# # import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.neural_network import MLPRegressor
#
# model = MLPRegressor()
# model.fit(X_train,Y_train)
# K_pred = model.predict(X_validation)
# score = r2_score(Y_validation, K_pred)
 #需要用测试数据检验这个模型好不好


# adaboost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

decisionTree1 = DecisionTreeClassifier()

decisionTree2 = DecisionTreeRegressor(max_leaf_nodes=10000000)
# model = AdaBoostClassifier(base_estimator=decisionTree1, n_estimators=50, learning_rate=0.05, random_state=0)
model = AdaBoostRegressor(base_estimator=decisionTree2, n_estimators=200, learning_rate=0.1, random_state=0, loss='square')


# model = AdaBoostRegressor(n_estimators=1000)
# model = XGBRegressor(n_estimators=1000)
model.fit(X_train,Y_train)



ypred = model.predict(X_validation)
print('MSLE of prediction on boston dataset:', mean_squared_log_error(Y_validation, ypred))
print('\n')

# scaler = MinMaxScaler(feature_range=(0,1))
# Xpred_stad = scaler.fit_transform(Xpred)
ypred2 = model.predict(Xpred)
list = []
for i in range (2000):
    templist = [i ,ypred2[i]]
    list.append(templist)
print(ypred2.shape)

column = ['Query ID', 'Predicted Cardinality']
test=pd.DataFrame(columns=column, data=list)
test.to_csv('test_output.csv', index=False)