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

### 1.3  我们的工作

// Todo

大概介绍一下我们使用了什么软件，做了什么事（编码、选模型、调参），取得了什么成果，还有什么地方可以展望。

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

## 4  总结与展望

1.3复述一遍，可以让AI写

## 参考文献

