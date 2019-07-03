# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:52:44 2019

Based on LogisticRegression.py
@author: Joel Wang
"""


"""
函数说明：

#定义一个logistic类

#1. 定义sigmod函数

#2. 定义fit拟合函数，包括计算损失J,损失梯度dJ,梯度下降计算参数theta的gradient_descent

#3. 预测，predict_prob计算概率（logistic函数值）以及predict计算类别

"""

import numpy as np
from sklearn.metrics import accuracy_score

#定义一个logistic类
class MyLogisticRegression(object):

    def __init__(self):
        #初始化参数
        self.coef=None
        self.intercept = None
        self._theta = None
#1. 定义sigmod函数
    def sigmod(self,t):
        return 1. / (1+np.exp(-t))

#2. 定义fit拟合函数，包括计算损失J,损失梯度dJ,梯度下降计算参数theta的gradient_descent
    def fit(self,X_train,y_train,eta=0.01,n_iters=1e4):
        #传入，训练集数据X_train，训练集标签Y_train,学习率eta,迭代次数niters
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        def J(theta, X_b, y):
            #传入当前计算的训练集数据X_b，训练集标签y
            y_hat = self.sigmod(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
            except:
                return float('inf')
            
        def dJ(theta, X_b, y):
            #参数同上，计算损失J对参数theta的梯度
            
            return X_b.T.dot(self.sigmod(X_b.dot(theta))-y) / len(y)
            
        def gradient_descent(X_b,y,initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            #梯度下降计算传入数据，标签，初始theta，学习率，代数，和代之间的误差
            theta = initial_theta
            cur_iter = 0
            
            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta =  theta - eta*gradient
                if abs(J(theta, X_b,y)-J(last_theta,X_b,y)) < epsilon:
                    break
                
                cur_iter+=1
                
            return theta
                
                
                
        #调用以上三个函数并计算：
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        #np.hstack是将两个矩阵按照水平方向叠加
        initial_theta = np.zeros(X_b.shape[1])
        #theta是一个增广系数theta_0+theata_1*x_1+theta_2*x_2+...
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        #截距项
        self.intercept = self._theta[0]
        #x_i前的系数
        self.coef = self._theta[1:]
        
        theta_mark=self._theta
        
        return self
#3. 预测，predict_prob计算概率（logistic函数值）以及predict计算类别
    def predict_proba(self, X_predict):
        """给定带预测的数据集X_predict,返回表示X_predict的结果概率向量"""
        assert self.intercept is not None and self.coef is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
        
        return self.sigmod(X_b.dot(self._theta))
    
    def predict(self, X_predict):
        """给定待预测数据集X_predict, 返回表示X_predict的结果向量"""
        assert self.intercept is not None and self.coef is not None, \
            "must fit before predict!"
        assert X_predict.shape[1]==len(self.coef), \
            "the feature number of X_predict must be equal to X_train"
        prob = self.predict_proba(X_predict)
        return np.array(prob >= 0.5, dtype='int')
    
    def score(self,X_test,y_test):
        #根据测试集X_test和y_test来确定当前模型的准确度，使用sklearn的
        
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    
    def __repr__(self):
        return "MyLogisticRegression()"
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    