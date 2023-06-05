# Design and Tuning of Query Scale Prediction Model Based on Weak Learners

# Completed by Yifan SHI & Yichen XU

## 1 Experimental Overview

### 1.1 Task Requirements

This experiment requires estimating the scale (cardinality) of query results using machine learning methods based on the given SQL queries and table features. The training set of SQL queries is provided in the `train.csv` file, and the table features are given in the `column_min_max_vals.csv` file. The test set that needs to be estimated is provided in the `test_without_label.csv` file.

The test set and training set are provided in comma-separated values (CSV) format. The following SQL statement:

<div style="page-break-after: always;"></div>

```sql
SELECT COUNT(*)
FROM Table_0 Alias_0, Table_1 Alias_1, ... # And Other Tables
WHERE Alias_k0.Column_t0 = Alias_k1.Column_t1 AND ... # And Other Join-Statements
AND Alias_r0.Column_s0 [op_0] Const_0 AND ... # And Other Condition Statements
```

is equivalent to the following CSV row:

```
Table_0 Alias_0,Table_1 Alias_1...#
Alias_k0.Column_t0=Alias_k1.Column_t1...#
Alias_r0.Column_s0,[op_0],Const_0...#
result
```

For example: `SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_type_id < 2`, the result is `1274246`, represented as `movie_companies mc##mc.company_type_id,<,2#1274246`.

Our goal is to train a machine learning model using 10,000 test data and provide scale predictions for 2,000 test data. The evaluation loss function is mean squared logarithmic error (MSLE).

### 1.2 Model Introduction

In this assignment, we use the XGBoost model.

XGBoost (Extreme Gradient Boosting) is a gradient boosting machine learning algorithm based on decision trees. It is suitable for classification and regression problems and is primarily used for supervised learning. Ensemble learning refers to building multiple classifiers (weak classifiers) to make predictions on a dataset and then combining the results of multiple classifiers using a certain strategy as the final prediction result. Therefore, ensemble learning effectively avoids the overfitting problem caused by a single learning model. XGBoost is such an efficient machine learning algorithm that improves the prediction accuracy of the model by building an ensemble of weak learners.

XGBoost has the following advantages:

1. It can automatically learn the importance of features and perform feature selection.
2. It is highly scalable and can use parallel computing to improve training speed.
3. It can optimize the performance of the model by adjusting hyperparameters.
4. It can handle high-dimensional and large-scale data.

We chose the XGBoost regressor as the final machine learning model because it automatically learns the importance of features, handles high-dimensional and large-scale data efficiently, and shows better performance than other machine learning algorithms in the SQL query dataset.

In conclusion, in this experiment, we chose the XGBoost regressor as our final machine learning model.

### 1.3 Our Work

#### 1.3.1 Experimental Environment

This experiment was conducted using the Jetbrains Pycharm integrated development platform with Python version 3.6.8. The following software packages were used:

* `scikit-learn`: Used for splitting the training set and validation set, specifying the loss function.
* `pandas`: Used for data frame operations.
* `numpy`: Used for

 basic data types.

* `math`: Provides basic mathematical functions.
* `xgb`: Provides the XGBoost framework.

The development environment is AMD Ryzen 9 7950X (32GB memory).

#### 1.3.2 Architecture Description

1. Perform data preprocessing and analysis on the given table data to understand the approximate distribution and scale of the data.
2. Encode the query conditions to obtain encoded vectors as the training set for input into the machine learning model for learning and iteration.
3. Choose a relatively good machine learning model and train it to predict the results of the test set.
4. Adjust the parameters of the selected model to improve the prediction accuracy.

#### 1.3.3 Results Obtained

On the testing website provided by the teaching assistant, we achieved good results with an MSLE of 0.9964944509395802 on the test set. The following image shows the score:

![score](/pics/score.png)

#### 1.3.4 Outlook

In this experiment, due to the lack of specific data, we can only use regression modeling to predict the scale of the given SQL query conditions. In the future, to improve the prediction accuracy, we can further improve the encoding method and try more machine learning models, neural networks, or even deep learning models that better match the test dataset, aiming to achieve better performance and provide users with a better experience.

## 2 Specific Implementation

### 2.1 Data Feature Analysis

After conducting preliminary analysis on the training and test datasets, we have reached the following conclusions:

- The independent variables to be predicted consist of three components: table information, join information, and condition information.
- There are a total of 6 tables present in the dataset.
- Considering the practical significance of join operations, there are 5 valid join operations present in the dataset.
- Considering the practical significance of the left-hand side of condition operations, there are 9 different data columns used as part of the conditions.
- All conditions are value-independent, meaning that the right-hand side of the conditions is always a constant.

### 2.2 Encoding Implementation

Based on the dataset features mentioned above, we encode the table information, join information, and condition information for each query separately and concatenate them into a final vector. This encoding method is similar to one-hot encoding, effectively consolidating and structuring all the relevant information from the original queries. The encoding process is as follows:

$$
Vec_{query}(q) = \mathbf{join} \{ \sigma_{table}(q), \sigma_{join}(q), \sigma_{condition}(q) \}
$$

The sub-vector for table information has a dimension of 6, where the value at dimension i indicates whether the i-th table, sorted by name, exists in the query:

$$
\sigma_{table}(q)[i]:=\text{Table}_i\in q
$$

The sub-vector for join information has a dimension of 5, where the value at dimension i indicates whether the i-th join condition, sorted by name, exists in the query:

$$
\sigma_{join}(q)[i]:=\text{Join}_i\in q
$$

The sub-vector for condition information has a dimension of 18, with each data column corresponding to 2 dimensions. We convert all the original conditions into an interval `[lower_bound, upper_bound]`, sort the 18 endpoint conditions by name, and define the normalized value at dimension i as follows:

$$
\sigma_{condition}(q)[i]:=scaler(transform(\text{Conditions}).sort()_i)\\
scaler(x):=\frac{x-\text{minVal}}{\text{maxVal}-\text{minVal}}
$$

For tables, joins, and lower bounds of conditions that do not exist in the query, the vector values are all set to 0. For upper bounds of conditions that do not exist in the query, the vector values are set to 1.

<div style="page-break-after: always;"></div>

### 2.3 Model Training

For the specific implementation, we primarily utilize the following modules and models from `sklearn` and `xgb`:

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

We use `train_test_split` to split the training dataset into a training set and a validation set. The evaluation metrics used for the validation set include `mean_squared_log_error` and `accuracy_score` to assess the model's performance under different parameter settings. We employ `GridSearchCV` for hyperparameter tuning.

Regarding model selection, we tested various classifiers and regressors, and the

 results were evident: this task is better suited for a regressor. In addition to the XGBoost model, we also experimented with KNN, AdaBoost, and Decision Tree models. After comparing the results, we ultimately selected the XGBoost regressor as the relatively superior choice.

Through multiple comparative experiments, we selected the XGBoost regressor with the best performance as the final model. After training with the provided samples, we predicted the test results and used the `pandas` library to output them to a result file for submission to the evaluation website.

## 3. Model Tuning and Evaluation

We split the entire training set into smaller training set $X_{train}$ and validation set $X_{validation}$ in a ratio of 1:0.05. By predicting on the validation set, we can adjust the model locally.

During the parameter tuning process, we pay special attention to the following parameters:

**XGBoost Framework Parameters (General parameters):**

* n_estimators [default = 100]
  * `n_estimators` is a crucial parameter that determines the complexity of our XGBoost model.
  * It represents the number of weak learners (decision trees) in our model. This parameter corresponds to `n_estimators` in sklearn's GBDT.
  * If `n_estimators` is too small, the model may underfit; if it's too large, the model may become overly complex. We need to choose a suitable value.
* objective [Regression default = reg:squarederror] [Multiclass default = multi:softmax]
  * `objective` specifies whether we are solving a classification, regression, or other type of problem, along with the corresponding loss function.
  * There are many possible values for this parameter, but we only focus on the ones used for classification and regression. For regression problems, we use `reg:squarederror`, which corresponds to mean squared error (MSE), and for multiclass problems, we use `multi:softmax`.

**XGBoost Weak Learner Parameters (Booster parameters):**

* learning_rate [default = 0.3]
  * `learning_rate` controls the shrinkage of weights for each weak learner, similar to `learning_rate` in sklearn's GBDT. A smaller `learning_rate` requires more iterations of weak learners.
  * Typically, we use both the learning rate and the maximum number of iterations (n_estimators) to determine the fitting effect of the algorithm. Therefore, these two parameters, n_estimators and learning_rate, need to be tuned together to achieve optimal results. Alternatively, we can fix the learning_rate first, tune n_estimators, then tune all other parameters, and finally readjust the learning_rate and n_estimators.
* max_depth [default = 6]
  * Controls the depth of the tree structure. If the data or features are small, this value may not be significant.
  * If the model has a large number of samples and features, it is necessary to limit the maximum depth. The specific value needs to be determined through grid search. This parameter corresponds to `max_depth` in sklearn's GBDT. Training deep trees with XGBoost consumes a large amount of memory.
* gamma [default = 0]
  * The threshold for loss reduction brought by the tree split in XGBoost.

Regarding parameter tuning, we use the method of grid search and cross-validation (GridSearchCV). In this process, we specify a range of parameters and adjust them step by step within the specified range. We train the learner using the adjusted parameters and find the parameters that achieve the highest accuracy on the validation set. It is an iterative process of training and comparing. $k$-fold cross-validation divides the entire dataset into $k$ subsets, and each time one subset is used as the test set while the remaining $k-1$ subsets are used as the training set. The model's score is calculated on the test set, and the final score is obtained by averaging the scores from $k$ iterations. GridSearchCV ensures that the best parameters are found within the specified parameter range.

The Python code for parameter tuning is shown below. Here, we demonstrate the process of adjusting the `learning_rate` parameter, and we ultimately choose `learning_rate = 0.11`, which results in the minimum loss.

```python
# xgboost parameter tuning
# cv_params = {'n_estimators': [567,577,587]} #577 is the best
# cv_params = {'max_depth': [4,5,6], 'min_child_weight': [3,4,5]} #5, 4 is the best
# cv_params = {'gamma': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]} #0 is the best

cv_params = {'learning_rate': [0.06,0.11,0.16]} #0.11 is the best
other_params = {'n_estimators':577,'learning_rate': 0.11}
model = XGBRegressor(**other_params)

optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)
evaluate_result = optimized_GBM.cv_results_['mean_test_score']
print('Each iteration result: {0}'.format(evaluate_result))
print('The best parameter values: {0}'.format(optimized_GBM.best_params_))
print('The best model score: {0}'.format(optimized_GBM.best_score_))
```

Each time, we choose different parameters or parameter combinations and input them into the `GridSearchCV` function provided by `sklearn` to obtain the local optimal solutions.

The table below shows the results of different parameters we tried, along with the corresponding loss values:

| Index | n_estimators | learning_rate | MSLE      |
| ----- | ------------ | ------------- | --------- |
| 1     | 567          | 0.06          | 1.017     |
| 2     | 567          | 0.11          | 0.997     |
| 3     | 567          | 0.16          | 1.010     |
| 4     | 577          | 0.06          | 1.017     |
| **5** | **577**      | **0.11**      | **0.996** |
| 6     | 577          | 0.16          | 1.010     |
| 7     | 587          | 0.06          | 1.017     |
| 8     | 587          | 0.11          | 0.996     |
| 9     | 587          | 0.16          | 1.011     |

After comparing the results, we obtained the relatively optimal model as shown below:

```python
model = XGBRegressor(n_estimators=577, learning_rate=0.11, max_depth=5, min_child_weight=4, scale_pos_weight=10, base_score=50)
```

## 4. Summary and Outlook

In this experiment, focusing on SQL queries and guided by estimating query cardinality, we have greatly developed our ability to model real-world problems, improved our abstract thinking, and deepened our understanding of machine learning methods taught in class and specific machine learning algorithms discussed after class. We manually implemented data mining, encoding, modeling, and analysis, which enhanced our understanding of the five "standard components" of machine learning algorithms and demonstrated their flexibility and power. In our future learning journey, we will continue to maintain an exploratory mindset and apply the ideas and methods learned from this experiment to solve practical problems in our lives.

## References

[1] Kipf, A., Kipf, T., Radke, B., Leis, V., Boncz, P., & Kemper, A. (2018). Learned cardinalities: Estimating correlated joins with deep learning. arXiv preprint arXiv:1809.00677.

[2] Dutt, A., Wang, C., Narasayya, V., & Chaudhuri, S. (2020). Efficiently approximating selectivity functions using low overhead regression models. Proceedings of the VLDB Endowment, 13(12), 2215-2228.

[3] Chen, T., He, T., Benesty, M., Khotilovich, V., Tang, Y., Cho, H., & Chen, K. (2015). Xgboost: extreme gradient boosting. R package version 0.4-2, 1(4), 1-4.

[4] Shehadeh, A., Alshboul, O., Al Mamlook, R. E., & Hamedat, O. (2021). Machine learning models for predicting the residual value of heavy construction equipment: An evaluation of modified decision tree, LightGBM, and XGBoost regression. Automation in Construction, 129, 103827.

[5] Park, Y., Zhong, S., & Mozafari, B. (2020, June). Quicksel: Quick selectivity learning with mixture models. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (pp. 1017-1033).

