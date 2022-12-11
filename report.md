# 基于弱学习器的查询规模预测模型设计与调优<br/>石依凡  2020202264<br/>徐一宸  2020201432

## 1  实验概述

### 1.1  任务要求

本实验需要针对给定的SQL查询和表特征，使用机器学习的方法，估算查询结果的规模（Cardinality）。SQL查询的训练集由`train.csv`文件给出，表特征由`column_min_max_vals.csv`文件给出，需要估算的测试集由`test_without_label.csv`文件给出。

测试集和训练集由逗号分隔值（CSV）格式给出。下面的SQL语句：

```sql
SELECT COUNT(*)
FROM Table_0 Alias_0, Table_1 Alias_1, ... # And Other Tables
WHERE Alias_k0.Column_t0 = Alias_k1.Column_t1 AND ... # And Other Join-Statements
AND Alias_r0.Column_s0 [op_0] Const_0 AND ... # And Other Condition Statements
```

等价于为下面的CSV行：

```
Table_0 Alias_0,Table_1 Alias_1...#
Alias_k0.Column_t0=Alias_k1.Column_t1...#
Alias_r0.Column_s0,[op_0],Const_0...#
result
```

例如：`SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_type_id < 2`，结果是`1274246`，就用`movie_companies mc##mc.company_type_id,<,2#1274246`表示。

我们的目标就是由$10000$条测试数据训练机器学习模型，并给出$2000$条测试数据的规模预测值。评估损失函数为均方对数误差（MSLE）函数。

### 1.2  模型介绍

// Todo
我们用到的模型是什么类型，什么特点，适用于什么情景，为什么选择这个模型。600字。（大而空的话）

在本次作业中，我们采取的是XGBoost的模型。 XGBoost是“Extreme Gradient Boosting”的简称， 
它是一种基于决策树的集成机器学习算法，使用梯度上升框架，适用于分类和回归问题，并且主要用来解决有监督学习问题。
所谓集成学习，是指构建多个分类器（弱分类器）对数据集进行预测，然后用某种策略将多个分类器预测的结果集成起来，作为最终预测结果。
因此，集成学习很好的避免了单一学习模型带来的过拟合问题.
XGBoost就是这样一种高效的机器学习算法，它通过构建弱学习器的集成，来提高模型的预测准确度。

XGBoost 具有以下优点：
1. 可以自动学习特征的重要性，并进行特征选择。
2. 具有高度的可扩展性，可以使用并行计算来提高训练速度。
3. 可以通过调整超参数来优化模型的性能。
4. 可以处理高维度和大规模的数据。

我们之所以使用XGBoost模型，是因为本次试验数据量较大，数据维数较多，相较于其他机器学习算法，XGBoost能够在SQL查询数据集中有着更好的表现，能够很好地预测
查询的Cardinality。在本次实验中，我们选择XGBoost回归器。


### 1.3  我们的工作

// Todo

大概介绍一下我们使用了什么软件，做了什么事（编码、选模型、调参），取得了什么成果，还有什么地方可以展望。

####  实验环境准备

在本次实验中，我们使用PyChram软件作为我们实现代码的编辑器。
PyCharm是一种Python IDE，带有一整套可以帮助用户在使用Python语言开发时提高其效率的工具。

##### 代码实现

1. 我们对给出的表数据做数据预处理和数据分析，以便得知数据的大致分布和规模。
2. 我们对查询条件进行编码，得到编码后的若干向量得到训练集，以便投入机器学习模型进行学习和迭代。
3. 我们选择一个效果相对较好的机器学习模型，把我们得到的训练集喂给该模型，预测出测试集的结果 
4. 我们会对选定的模型进行参数调整，以便让模型能够在测试集上有更好的表现，提升模型的的预测准确度。

####  取得的成果

在助教提供的网站上，我们取得了较好的效果，测试集的MSLE达到了1.0514347331312608

[jpg]

####  未来展望

在本次实验中，由于没有具体的数据，我们只能通过建模回归的方法来对给出的SQL查询条件进行规模预测。未来如果想要让预测的准确率更好，我们应进一步改进编码模式，尝试更多的、与测试集数据更加匹配的机器学习、神经网络甚至深度学习模型，争取能够达到更好的表现，给用户更加良好的体验。

## 2  具体实现

### 2.1  数据集分析

// Todo

训练、测试集的特征是什么？（参考gdy的Jupyter）

### 2.2  编码实现

// Todo

怎么编码（类似论文上的一张图）

怎么标准化

### 2.3  模型训练

// Todo

使用到的库，比较过的模型（细致的说明我们一步步选择这个模型的过程）

在具体实现上，我们主要用到了sklearn相关的模型：
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, accuracy_score, mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
```

我们采用`train_test_split`来对训练集进行分割，
采用`mean_squared_log_error`和`accuracy_score`来对验证集进行评分，从而判断在当前参数下模型的优秀程度。

在模型的选用上，我们对各种模型的分类器和回归器都进行了测试，结果是显然的——本次实验更加适用于回归器。
除了XBGoost模型之外，我们还尝试了KNN模型、AdaBoost模型以及决策树模型，

【他们的效果如下？】（这里用把所有的都跑一遍吗？

我们通过多次对比实验，选取表现更好的XGBoost回归器作为最终的模型。

## 3  调优与评估

为什么这样训练（分析参数原因）

再次划分测试集、验证集，交叉验证

给出不同参数下的（交叉验证/验证集）损失结果（表格）

| 序号 | 参数1 | 参数2 | 损失 |
| ---- | ----- | ----- | ---- |
|      |       |       |      |
|      |       |       |      |
|      |       |       |      |
|      |       |       |      |
|      |       |       |      |

我们把整个训练集按照1:0.05的比例划分成更小的训练集X_train和验证集X_validation。我们通过对验证集进行预测的到的结果就可以在本地来调整模型。

在调参过程中，我们着重关注的参数有以下这些：

XGBoost框架参数(General parameters):
1. booster [ default = gbtree ]<br/>
booster决定了XGBoost使用的弱学习器类型，可以是默认的gbtree, 也就是CART决策树，还可以是线性弱学习器gblinear以及DART。一般来说，我们使用gbtree就可以了，不需要调参。
2. n_estimators [ default = 100 ]<br/>
n_estimators则是非常重要的要调的参数，它关系到我们XGBoost模型的复杂度，因为它代表了我们决策树弱学习器的个数。这个参数对应sklearn GBDT的n_estimators。n_estimators太小，容易欠拟合，n_estimators太大，模型会过于复杂，一般需要调参选择一个适中的数值。
3. objective [ 回归default = reg:squarederror ][ 二分类default = binary:logistic ][ 多分类default = multi:softmax ]<br/>
objective代表了我们要解决的问题是分类还是回归，或其他问题，以及对应的损失函数。具体可以取的值很多，一般我们只关心在分类和回归的时候使用的参数。在回归问题objective一般使用reg:squarederror ，即MSE均方误差。二分类问题一般使用binary:logistic, 多分类问题一般使用multi:softmax。

XGBoost 弱学习器参数(Booster parameters):
1. learning_rate [default = 0.3 ]<br/>
learning_rate控制每个弱学习器的权重缩减系数，和sklearn GBDT的learning_rate类似，较小的learning_rate意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参才有效果。当然也可以先固定一个learning_rate ，然后调完n_estimators，再调完其他所有参数后，最后再来调learning_rate和n_estimators。
2. max_depth [ default = 6 ]<br/>
控制树结构的深度， 数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，需要限制这个最大深度， 具体的取值一般要网格搜索调参。这个参数对应sklearn GBDT的max_depth。要注意，在训练深树时，XGBoost会大量消耗内存。
3. gamma [ default = 0 ]<br/>
XGBoost的决策树分裂所带来的损失减小阈值。也就是我们在尝试树结构分裂时，会尝试最大数下式：
4. reg_alpha[ default = 0 ]/reg_lambda[ default = 1]<br/>
这2个是XGBoost的正则化参数。reg_alpha是L1正则化系数，reg_lambda是L2正则化系数，
5. subsample [ default = 1]<br/>
子采样参数，这个也是不放回抽样，和sklearn GBDT的subsample作用一样。也就是说随机森林的每棵树的训练有可能会碰到训练集中两个样本重合的状况，因为是放回抽样；像GBDT\XGBoost训练每棵树的训练样本是不会重合的，只是subsample会控制整个训练集被选中的概率。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。初期可以取值1，如果发现过拟合后可以网格搜索调参找一个相对小一些的值。
6. colsample_bytree, colsample_bylevel, colsample_bynode [ default = 1 ]<br/>
这三个参数都是用于特征采样的，默认都是不做采样，即使用所有的特征建立决策树。colsample_bytree控制整棵树的特征采样比例，colsample_bylevel控制某一层(depth)的特征采样比例，而colsample_bynode(split)控制某一个树节点的特征采样比例。比如我们一共64个特征，则假设colsample_bytree，colsample_bylevel和colsample_bynode都是0.5，则某一个树节点分裂时会随机采样8个特征来尝试分裂子树。

最终我们的模型选择的参数如下: 
model = XGBRegressor(n_estimators=457, learning_rate=0.18, max_depth=5, gamma=0.2)，其余参数都为默认值。

## 4  总结与展望

1.3复述一遍，可以让AI写

## 参考文献

