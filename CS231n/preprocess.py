#coding=utf-8
import numpy as np
X=np.array([[1,3,3,5],[2,4,5,6],[3,8,0,9]])
#预处理的2个方法
#data shape
#X--NxD，N为样本数，为axis=0, D为特征维数，为axis=1

#1 普通预处理: 中心化+归一化
#均值减法：
X-=np.mean(X,axis=0) #减去每一列（所有样本的同一个特征）的均值

#归一化：
X /= np.std(X,axis=0) #除以每一列的0中心标准差


#2 PCA+白化
#PCA降维
#0中心化
X-=np.mean(X,axis=0)
#协方差
cov=np.dot(X.T,X) / X.shape[0]
#奇异值分解
U,S,V = np.linalg.svd(cov)
#去相关性
Xrot = np.dot(X,U)
#取特征值较大的2维
Xrot_reduced = np.dot(X,U[:,:100])

#白化操作
#除以特征值,1e-5防止特征值为0
Xwhite = Xrot / np.sqrt(S+1e-5)
