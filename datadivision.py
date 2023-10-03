# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:50:19 2021

@author: dell
"""

import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

hyperparams = [60000, 784, 10, 1e-3, 20, 'MNIST', 1e-3]
train_dataset = dsets.MNIST(root = './data/mnist', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = dsets.MNIST(root = './data/mnist', train = False, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)

import pickle
for i, (images, labels) in enumerate(train_loader):
    pickle.dump(images[0:25000,:,:,:], open(f"images_data1.p", "wb"))
    pickle.dump(labels[0:25000], open(f"labels_data1.p", "wb"))
    pickle.dump(images[25000:50000,:,:,:], open(f"images_data2.p", "wb"))
    pickle.dump(labels[25000:50000], open(f"labels_data2.p", "wb"))
    pickle.dump(images[50000:55000,:,:,:], open(f"images_data3.p", "wb"))
    pickle.dump(labels[50000:55000], open(f"labels_data3.p", "wb"))
    pickle.dump(images[50000:55000,:,:,:], open(f"images_val.p", "wb"))
    pickle.dump(labels[50000:55000], open(f"labels_val.p", "wb"))

for i, (images, labels) in enumerate(test_loader):
    pickle.dump(images[0:10000,:,:,:], open(f"test_data.p", "wb"))
    pickle.dump(labels[0:10000], open(f"labels_test.p", "wb"))



'''
import numpy as np 
import pandas as pd
 
trainData = pd.read_csv("mnist_train.csv").values
train_data = trainData[0:60000, 1:]  
train_label = trainData[0:60000, 0]  # 50000 代表样本数
testData = pd.read_csv("mnist_test.csv").values
test_data = testData[0:10000, 1:]  
test_label = testData[0:10000, 0]     # 10000 代表标签数


train_data1 = trainData[0:25000, 1:]  
train_label1 = trainData[0:25000, 0]  # 50000 代表样本数
train_data2 = trainData[25000:50000, 1:]  
train_label2 = trainData[25000:50000, 0]  # 50000 代表样本数
train_data3 = trainData[50000:55000, 1:]  
train_label3 = trainData[50000:55000, 0]  # 50000 代表样本数
Val_data = trainData[55000:60000, 1:]  
Va_label = trainData[55000:60000, 0]  # 50000 代表样本数

 
np.savetxt('train_data1.csv',train_data1) 
np.savetxt('train_label1.csv',train_label1) 

np.savetxt('test_data.csv',train_data1) 
np.savetxt('test_label.csv',train_label1) 

'''





