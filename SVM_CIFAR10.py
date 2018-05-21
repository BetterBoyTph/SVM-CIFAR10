# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:55:19 2018

@author: zly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM
from PIL import Image
from sklearn import svm as s
import time

def unpickle(file):
    """
    功能：将CIFAR10中的数据转化为字典形式
    
    （1）加载data_batch_i(i=1,2,3,4,5)和test_batch文件返回的字典格式为：
    dict_keys([b'filenames', b'data', b'labels', b'batch_label'])
    其中每一个batch中：
    dict[b'data']为（10000,3072）的numpy array
    dict[b'labels']为长度为10000的list
    
    （2）加载batchs.meta文件返回的字典格式为：
    dict_keys([b'num_cases_per_batch', b'num_vis', b'label_names'])
    其中dict[b'label_names']为一个list，记录了0-9对应的类别为：
    [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
    """
    
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    

def load_CIFAR10():
    """
    功能：从当前路径下读取CIFAR10数据
    
    输出：
    -x_train:(numpy array)训练样本数据(N,D)
    -y_train:(numpy array)训练样本数标签(N,)
    -x_test:(numpy array)测试样本数据(N,D)
    -y_test:(numpy array)测试样本数标签(N,)
    """
    x_t=[]
    y_t=[]
    for i in range(1,6):
        path_train=os.path.join('cifar-10-batches-py','data_batch_%d'%(i))
        data_dict=unpickle(path_train)
        x=data_dict[b'data'].astype('float')
        y=np.array(data_dict[b'labels'])
        
        x_t.append(x)
        y_t.append(y)
        
    #将数据按列堆叠进行合并,默认按列进行堆叠
    x_train=np.concatenate(x_t)
    y_train=np.concatenate(y_t)
    
    path_test=os.path.join('cifar-10-batches-py','test_batch')
    data_dict=unpickle(path_test)
    x_test=data_dict[b'data'].astype('float')
    y_test=np.array(data_dict[b'labels'])
    
    return x_train,y_train,x_test,y_test
    
def data_processing():
    """
    功能：进行数据预处理

    输出：
    x_tr:(numpy array)训练集数据
    y_tr:(numpy array)训练集标签
    x_val:(numpy array)验证集数据
    y_val:(numpy array)验证集标签
    x_te:(numpy array)测试集数据
    y_te:(numpy array)测试集标签   
    x_check:(numpy array)用于梯度检查的子训练集数据
    y_check:(numpy array)用于梯度检查的子训练集标签
    """
    
    #加载数据
    x_train,y_train,x_test,y_test=load_CIFAR10()
    
    num_train=10000
    num_test=1000
    num_val=1000
    num_check=100
    
    #创建训练样本
    x_tr=x_train[0:num_train]
    y_tr=y_train[0:num_train]

    #创建验证样本
    x_val=x_train[num_train:(num_train+num_val)]
    y_val=y_train[num_train:(num_train+num_val)]

    #创建测试样本
    x_te=x_test[0:num_test]
    y_te=y_test[0:num_test]

    #从训练样本中取出一个子集作为梯度检查的数据
    mask=np.random.choice(num_train,num_check,replace=False)
    x_check=x_tr[mask]
    y_check=y_tr[mask]

    #计算训练样本中图片的均值
    mean_img=np.mean(x_tr,axis=0)
    
    #所有数据都减去均值做预处理
    x_tr+=-mean_img
    x_val+=-mean_img
    x_te+=-mean_img
    x_check+=-mean_img
    
    #加上偏置项变成(N,3073)
    #np.hstack((a,b))等价于np.concatenate((a,b),axis=1),在横向合并
    #np.vstack((a,b))等价于np.concatenate((a,b),axis=0)，在纵向合并
    x_tr=np.hstack((x_tr,np.ones((x_tr.shape[0],1))))
    x_val=np.hstack((x_val,np.ones((x_val.shape[0],1))))
    x_te=np.hstack((x_te,np.ones((x_te.shape[0],1))))
    x_check=np.hstack((x_check,np.ones((x_check.shape[0],1))))
    
    return x_tr,y_tr,x_val,y_val,x_te,y_te,x_check,y_check

def VisualizeWeights(best_W):
    #去除最后一行偏置项
    w=best_W[:-1,:]
    w=w.T
    w=np.reshape(w,[10,3,32,32])
    #对应类别
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes=len(classes)
    plt.figure(figsize=(12,8))
    for i in range(num_classes):
        plt.subplot(2,5,i+1)
        #将图像拉伸到0-255
        x=w[i]
        minw,maxw=np.min(x),np.max(x)
        wimg=(255*(x.squeeze()-minw)/(maxw-minw)).astype('uint8')
        
        r=Image.fromarray(wimg[0])
        g=Image.fromarray(wimg[1])
        b=Image.fromarray(wimg[2])
        #合并三通道
        wimg=Image.merge("RGB",(r,g,b))
#        wimg=np.array(wimg)
        plt.imshow(wimg)
        plt.axis('off')
        plt.title(classes[i])
    
    
    
    
#主函数
if __name__ == '__main__':
    #进行数据预处理
    x_train,y_train,x_val,y_val,x_test,y_test,x_check,y_check= data_processing()
    
#    在进行交叉验证前要可视化损失函数值的变化过程，检验训练过程编程是否正确
#    svm=SVM()
#    history_loss=svm.train(x_train,y_train,reg=1e5,learning_rate=1e-7,num_iters=1500,batch_size=200,verbose=True)
#    plt.figure(figsize=(12,8))
#    plt.plot(history_loss)
#    plt.xlabel('iteration')
#    plt.ylabel('loss')
#    plt.show()
    
    start=time.clock()
    #使用验证集调参
    learning_rate=[7e-6,1e-7,3e-7]
    regularization_strength=[1e4,3e4,5e4,7e4,1e5,3e5,5e5]

    max_acc=-1.0
    for lr in learning_rate:
        for rs in regularization_strength:
            svm=SVM()
            #训练
            history_loss=svm.train(x_train,y_train,reg=rs,learning_rate=lr,num_iters=2000)
            #预测验证集类别
            y_pre=svm.predict(x_val)
            #计算验证集精度
            acc=np.mean(y_pre==y_val)
            
            #选取精度最大时的最优模型
            if(acc>max_acc):
                max_acc=acc
                best_learning_rate=lr
                best_regularization_strength=rs
                best_svm=svm
                
            print("learning_rate=%e,regularization_strength=%e,val_accury=%f"%(lr,rs,acc))
    print("max_accuracy=%f,best_learning_rate=%e,best_regularization_strength=%e"%(max_acc,best_learning_rate,best_regularization_strength))
    end=time.clock()

    #用最优svm模型对测试集进行分类的精度
    #预测测试集类别
    y_pre=best_svm.predict(x_test)
    #计算测试集精度
    acc=np.mean(y_pre==y_test)
    print('The test accuracy with self-realized svm is:%f'%(acc))
    print("\nProgram time of self-realized svm is:%ss"%(str(end-start)))
    
    #可视化学习到的权重
    VisualizeWeights(best_svm.W)
    
    #使用自带的svm进行分类
    start=time.clock()
    lin_clf = s.LinearSVC()
    lin_clf.fit(x_train,y_train)
    y_pre=lin_clf.predict(x_test)
    acc=np.mean(y_pre==y_test)
    print("The test accuracy with svm.LinearSVC is:%f"%(acc))
    end=time.clock()
    print("Program time of svm.LinearSVC is:%ss"%(str(end-start)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    