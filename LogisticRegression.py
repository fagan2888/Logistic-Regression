# coding:utf-8
"""
作者：zhaoxingfeng	日期：2017.04.19
版本：2.0
功能：Logistic回归，Logistic Regression (LR)
参考文献:
[1]洞庭小哥.【机器学习笔记1】Logistic回归总结[DB/OL].http://blog.csdn.net/dongtingzhizi/article/details/15962797,2013-11-13.
[2]PETER HARRINGTON. Machine Learning in Action［M].MANNING Shelter Island, 2012
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
pd.set_option('chained_assignment', None)
from sklearn.cross_validation import train_test_split

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# 梯度上升，对所有数据进行计算，然后更新权值
def gradAscent(data):
    dataRaw = np.mat(data)
    dataMat = dataRaw[:, :-1]
    labelMat = dataRaw[:, -1]
    feature = dataMat.shape[1]
    alpha = 0.001
    maxIter = 500
    weights = np.ones((feature, 1))
    for i in range(maxIter):
        y = sigmoid(dataMat * weights)
        erro = labelMat - y
        weights += alpha * dataMat.transpose() * erro
    return weights

# 画散点图和分类线
def plotFit(data, weights):
    data1 = data[data.label == 1][['x1', 'x2']].values
    data0 = data[data.label == 0][['x1', 'x2']].values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data1[:, 0], data1[:, 1], s=30, c='red', marker='o')
    ax.scatter(data0[:, 0], data0[:, 1], s=30, c='green', marker='s')
    # 分类线上w*x=0，知道x1,即可求出x2
    x1 = np.arange(-4, 4, 0.1)
    x2 = (-weights[0] - weights[1] * x1) / weights[2]
    ax.plot(x1, x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升，每次只用一条数据更新权值
def gradAscentRandom(data, maxIter=150):
    dataRaw = np.mat(data)
    dataMat = dataRaw[:, :-1]
    labelMat = dataRaw[:, -1]
    num, feature = dataMat.shape
    weights = np.ones((feature, 1))
    alpha = 1
    for i in range(maxIter):
        dataIndex = range(num)
        for j in range(num):
            randIndex = random.sample(dataIndex, 1)[0]
            dataIndex.remove(randIndex)
            y = sigmoid(dataMat[randIndex] * weights)[0, 0]
            erro = labelMat[randIndex, 0] - y
            weights += alpha * dataMat[randIndex].transpose() * erro
    return weights

# 测试算法
def test():
    df = pd.read_csv('testSet.txt', header=None, sep='\t', names=['x1', 'x2', 'label'])
    df['x0'] = 1
    df = df[['x0', 'x1', 'x2', 'label']]
    # weights = gradAscent(df.values)
    weights = gradAscentRandom(df.values)
    print("weights", weights)
    plotFit(df, weights)
# test()

# 记录运行时间
def runTime(func):
    def wrapper(*args, **kwargs):
        import time
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print "{0} runTime: {1:.2f}s".format(func.__name__, t2 - t1)
    return wrapper

# 分类马疝气病数据集
def horseTest():
    trainData = pd.read_csv('horseColicTraining.txt', header=None, sep='\t').values
    testData = pd.read_csv('horseColicTest.txt', header=None, sep='\t').values
    weights = gradAscentRandom(trainData, 300)
    inX = np.mat(testData[:, :-1]) * weights
    predict = [1 if sigmoid(x[0, 0]) > 0.5 else 0 for x in inX]
    accuracy = list((testData[:, -1] - predict)).count(0) / len(testData)
    return accuracy

# 分类马疝气病数据集，运行多次求准确率均值
@runTime
def multiHorse():
    numTests = 1
    accuracyList = [horseTest() for i in range(numTests)]
    accuracy = sum(accuracyList) / numTests
    print("accuracy = {}%".format(accuracy * 100))
multiHorse()

# 采用sklearn库中的logregression
def sklearnHorse():
    from sklearn.linear_model import LogisticRegression
    train = pd.read_csv('horseColicTraining.txt', header=None, sep='\t').values
    test = pd.read_csv('horseColicTest.txt', header=None, sep='\t').values
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]
    classifier = LogisticRegression()
    classifier.fit(train_x, train_y)
    accuracy = list(classifier.predict(test_x) - test_y).count(0) / len(test_y)
    print("accuracy = {}%".format(accuracy * 100))
sklearnHorse()

print("******************************************")

# 分类印第安人糖尿病数据集
@runTime
def pimaTest():
    data = pd.read_csv('pima indians.txt', header=None, sep=',').values
    trainData, testData, train_y, test_y = train_test_split(data, data, test_size=1/3)
    weights = gradAscentRandom(trainData, 300)
    inX = np.mat(testData[:, :-1]) * weights
    predict = [1 if sigmoid(x[0, 0]) > 0.5 else 0 for x in inX]
    accuracy = list((testData[:, -1] - predict)).count(0) / len(testData)
    print("accuracy = {}%".format(accuracy * 100))
pimaTest()

# 采用sklearn库中的logregression
def sklearnPima():
    from sklearn.linear_model import LogisticRegression
    data = pd.read_csv('pima indians.txt', header=None, sep=',').values
    train_x, test_x, train_y, test_y = train_test_split(data[:, :-1], data[:, -1], test_size=1/3)
    classifier = LogisticRegression()
    classifier.fit(train_x, train_y)
    accuracy = list(classifier.predict(test_x) - test_y).count(0) / len(test_y)
    print("accuracy = {}%".format(accuracy * 100))
sklearnPima()
