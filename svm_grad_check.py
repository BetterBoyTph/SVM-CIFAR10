# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:53:51 2018

@author: tph
"""

#==============================================================================
# 对SVM做梯度检查
#==============================================================================

import numpy as np
from SVM_CIFAR10 import *

def svm_loss_naive(w,x,y,reg):
    
    """
    功能：非矢量化版本的损失函数
    
    输入：
    -w:(numpy array)权重(3073,10)
    -x：(numpy array)样本数据（N,D)
    -y：(numpy array)标签（N，）
    -reg：(float)正则化强度
    
    输出：
    (float)损失函数值loss
    (numpy array)权重分析梯度dW
    """
    num_train,dim=x.shape
    num_class=10#w.shape[1]

    #初始化
    loss=0.0
    dW=np.zeros((dim,num_class))

    for i in range(num_train):
        scores=x[i].dot(w)
        #计算边界,delta=1
        margin=scores-scores[y[i]]+1
        #把正确类别的归0
        margin[y[i]]=0

        for j in range(num_class):
            #max操作
            if j==y[i]:
                continue
            if margin[j]>0:
                loss+=margin[j]
                dW[:,y[i]]+=-x[i]
                dW[:,j]+=x[i]

    #要除以N
    loss/=num_train
    dW/=num_train
    #加上正则项
    loss+=0.5*reg*np.sum(w*w)
    dW+=reg*w

    return loss,dW

def svm_loss_vectorized(w, x, y, reg):
    """
    功能：矢量化版本的损失函数
    
    输入：
    -x：(numpy array)样本数据（N,D)
    -y：(numpy array)标签（N，）
    -reg：(float)正则化强度
    
    输出：
    (float)损失函数值loss
    (numpy array)权重梯度dW
    """
    loss=0.0
    dW=np.zeros(w.shape)

    num_train=x.shape[0]
    scores=x.dot(w)
    margin=scores-scores[np.arange(num_train),y].reshape(num_train,1)+1
    margin[np.arange(num_train),y]=0.0
    #max操作
    margin=(margin>0)*margin
    loss+=margin.sum()/num_train
    #加上正则化项
    loss+=0.5*reg*np.sum(w*w)

    #计算梯度
    margin=(margin>0)*1
    row_sum=np.sum(margin,axis=1)
    margin[np.arange(num_train),y]=-row_sum
    dW=x.T.dot(margin)/num_train+reg*w

    return loss,dW
    
    
    
    
    
    
def svm_grad_check(f,w,analytic_grad,num_checks=10,h=1e-5):
    """
    功能：进行梯度检查，查看分析梯度和数值梯度是否满足一定的误差要求
    
    输入：
    f:计算损失值的函数
    w:权重,f(w)表示损失值
    analytic_grad:分析梯度
    num_checks:检查点数目
    h:求数值梯度的值的h
    """
    
    for i in range(num_checks):
        #随机选梯度矩阵中的一个坐标位置(row,col)
        row=np.random.randint(w.shape[0])
        col=np.random.randint(w.shape[1])  
        oldval=w[row,col]

        #计算f(x+h)
        w[row,col]=oldval+h
        fxph=f(w)
       
        #计算f(x-h)
        w[row,col]=oldval-h
        fxsh=f(w)
        
        #计算该点数值梯度
        grad_numerical=(fxph-fxsh)/(2*h)
        
        #还原改点值
        w[row,col]=oldval
        
        #计算该点的分析梯度
        grad_analytic=analytic_grad[row,col]
        grad_analytic1=analytic_grad1[row,col]
        #计算误差率
        error=abs(grad_analytic-grad_numerical)/(abs(grad_analytic)+abs(grad_numerical))
        
        print("grad_numerical:%.10f,grad_analytic=%.10f,error=%e"%(grad_numerical,grad_analytic,error))
        if(error<1e-7):
            print('误差合理')
        


if __name__=='__main__':
    #创建用于梯度检查的样本x_check,y_check
    x_tr,y_tr,x_val,y_val,x_te,y_te,x_check,y_check= data_processing()
    
    #初始化权重
    W=0.005*np.random.randn(3073,10)
    
    #设置reg=0.0,忽略正则项
    print('忽略正则项：(error<1e-7为合理结果)')
    loss,analytic_grad=svm_loss_vectorized(W,x_check,y_check,reg=0.0)
    f=lambda w:svm_loss_vectorized(w,x_check,y_check,reg=0.0)[0]
    svm_grad_check(f,W,analytic_grad)
    
    print('\n加上正则项：(error<1e-7为合理结果)')
    #设置reg=1e2,加上正则项
    loss,analytic_grad=svm_loss_naive(W,x_check,y_check,reg=1e2)
    f=lambda w:svm_loss_naive(w,x_check,y_check,reg=1e2)[0]
    svm_grad_check(f,W,analytic_grad)



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        