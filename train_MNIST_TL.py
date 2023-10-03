import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from SNN import SNN
import time
import os
import argparse
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd



parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-gpu', type = int, default = 0)
parser.add_argument('-seed', type = int, default = 20)
parser.add_argument('-nps', type = str, default = 'C1')
parser.add_argument('-dts', type = str, default = 'C10')

opt = parser.parse_args()

#torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
#torch.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
num_epoch = 10#    80

find = False

test_scores = []
train_scores = []
ccost_scores = []
if_tb = False




import matplotlib.pyplot as plt

 


imag = pd.read_pickle(r'images_data1.p')
labe = pd.read_pickle(r'labels_data1.p')

imag2 = pd.read_pickle(r'images_data2.p')
labe2 = pd.read_pickle(r'labels_data2.p')

imag3 = pd.read_pickle(r'images_data3.p')
labe3 = pd.read_pickle(r'labels_data3.p')

test_data = pd.read_pickle(r'test_data.p')
labels_test = pd.read_pickle(r'labels_test.p')







hyperparams=[100, 784, 10, 0.005, 20, 'MNIST', 0.001]

#from dataset import loaddata
#writer, hyperparams, train_dataset, test_dataset, train_loader, test_loader = loaddata(opt.dts, if_tb)

writer = None
if if_tb:
    writer = SummaryWriter(comment = '-Mni')

hyperparams.append(opt.nps)
task = hyperparams[5]

path = '../dynamic_data/10_' + opt.dts + str(opt.seed) + '_' + str(hyperparams[3]) + '_' + str(hyperparams[6]) # 12 - 8
place = path + '/'

if find:
    if not os.path.exists(path):
        os.mkdir(path)

print('Dataset: ' + task)
print('Random Seed: {}'.format(opt.seed))
print('Length of Training Dataset: {}'.format(len(imag)))
print('Length of Test Dataset: {}'.format(len(test_data)))
print('Build Model')
print('Params come from ' + hyperparams[-1])




loss_val_8times=list() 
experiments=1 # 实验次数
for j in range(experiments):
        
    model = SNN(hyperparams)
 
    loss_function = nn.MSELoss()
    
    for i in range(model.len):
        if find:
            layer = model.layers[i]
            np.save(place + 'layer' + str(i) + '_a0', layer.a.detach().cpu().numpy())
            np.save(place + 'layer' + str(i) + '_b0', layer.b.detach().cpu().numpy())
            np.save(place + 'layer' + str(i) + '_c0', layer.c.detach().cpu().numpy())
            np.save(place + 'layer' + str(i) + '_d0', layer.d.detach().cpu().numpy())
    
    paras = dict(model.named_parameters())
    paras_new = []
    for k, v in paras.items():
        if k[9] == 'f':
            paras_new.append({'params': [v], 'lr': hyperparams[3]})
        else:
            paras_new.append({'params': [v], 'lr': hyperparams[6]})
    
    optimizer = torch.optim.Adam(paras_new)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma = 0.9)
    cossim = torch.nn.CosineSimilarity(dim = 1, eps = 1e-8)
    sigmoid = torch.nn.Sigmoid()
    norm = torch.nn.BatchNorm2d(1)
    
    
    
    
    #        model3=model
      #               torch.save(model3, 'model3.pt')
      #               model = torch.load("model2.pt")
    
    #model = torch.load("model1.pt")
    
    loss_val=list()
    for epoch in range(num_epoch):
        model.train()
        scheduler.step()
        print('Train Epoch ' + str(epoch + 1))
        start_time = time.time()
        total_loss = 0
        
        index = [i for i in range(test_data.shape[0])]
        np.random.shuffle(index) # 打乱索引
        test_data = test_data[index]
        test_data=torch.Tensor(test_data)
        labels_test = labels_test[index]
        
        for i in range (1, int(len(imag)/hyperparams[0])):
            if True:
                images=imag[(i-1)*hyperparams[0]:i*hyperparams[0],:,:,:]
                labels=labe[(i-1)*hyperparams[0]:i*hyperparams[0]]
            # if i < 60:
                if 1 == 1:
                    optimizer.zero_grad()
                    if images.type() == 'torch.DoubleTensor':
                        images = images.to(torch.float32)
                    images = Variable(images)
                    if task == 'MNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'FashionMNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'NETtalk':
                        labels = labels.float()
                        labels = Variable(labels)
                    elif task == 'Cifar10':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'NMNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'TiDigits':
                        labels = labels.long()
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'Timit':
                        images = norm(images.unsqueeze(1))
                        images = images.squeeze(1)
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
    
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    total_loss += float(loss)
                    loss.backward(retain_graph = True)
                    optimizer.step()
    #                print('Train loss',float(loss))
                    
                    '''
                    #Valiation
                    total_loss1 = 0
                    for i in range (1, int(len(test_data)/hyperparams[0])):
                        images1=test_data[(i-1)*hyperparams[0]:i*hyperparams[0],:]
                        labels1=labels_test[(i-1)*hyperparams[0]:i*hyperparams[0]]
                        one_hot1 = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels1.unsqueeze(1), 1)
                        labels1 = Variable(one_hot1)
                        outputs1 = model(images1)
                        loss1 = loss_function(outputs1, labels1)
                        total_loss1 += float(loss1)
                    print('Val loss',total_loss1/i)
                    '''
        
        
    
        for i in range (1, int(len(imag2)/hyperparams[0])):
            if True:
                images=imag2[(i-1)*hyperparams[0]:i*hyperparams[0],:,:,:]
                labels=labe2[(i-1)*hyperparams[0]:i*hyperparams[0]]
            # if i < 60:
                if 1 == 1:
                    optimizer.zero_grad()
                    if images.type() == 'torch.DoubleTensor':
                        images = images.to(torch.float32)
                    images = Variable(images)
                    if task == 'MNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'FashionMNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'NETtalk':
                        labels = labels.float()
                        labels = Variable(labels)
                    elif task == 'Cifar10':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'NMNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'TiDigits':
                        labels = labels.long()
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'Timit':
                        images = norm(images.unsqueeze(1))
                        images = images.squeeze(1)
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
    
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    total_loss += float(loss)
                    loss.backward(retain_graph = True)
                    optimizer.step()
    #                print('loss',loss)
                        
    
        for i in range (1, int(len(imag3)/hyperparams[0])):
            if True:
                images=imag3[(i-1)*hyperparams[0]:i*hyperparams[0],:,:,:]
                labels=labe3[(i-1)*hyperparams[0]:i*hyperparams[0]]
            # if i < 60:
                if 1 == 1:
                    optimizer.zero_grad()
                    if images.type() == 'torch.DoubleTensor':
                        images = images.to(torch.float32)
                    images = Variable(images)
                    if task == 'MNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'FashionMNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'NETtalk':
                        labels = labels.float()
                        labels = Variable(labels)
                    elif task == 'Cifar10':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'NMNIST':
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'TiDigits':
                        labels = labels.long()
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
                    elif task == 'Timit':
                        images = norm(images.unsqueeze(1))
                        images = images.squeeze(1)
                        one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                        labels = Variable(one_hot)
    
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    total_loss += float(loss)
                    loss.backward(retain_graph = True)
                    optimizer.step()
    #                print('loss',loss)
                        
    
        

                         
        if_test=1
        model.eval()
        correct = 0
        total = 0
        loss1=list()
        if if_test:
            print('Test Epoch ' + str(epoch + 1))
            test_or_train = 'test'
        else:
            test_or_train = 'train'
    
    
        if task == 'MNIST':
            for i in range (1, int(len(test_data)/hyperparams[0])):
                images=test_data[(i-1)*hyperparams[0]:i*hyperparams[0],:,:,:]
                labels=labels_test[(i-1)*hyperparams[0]:i*hyperparams[0]]
                images = Variable(images)
                labels = Variable(labels)
                outputs = model(images)
                
                one_hot1 = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                labels1 = Variable(one_hot1)
                loss = loss_function(outputs, labels1)
                loss1=np.hstack((loss1,float(loss)))
                
                total += labels.size(0)
                pred = outputs.max(1)[1]
                correct += (pred == labels).sum()
            correct = correct.item()
        loss_val=np.hstack((loss_val,loss1))
    
    
        ccost = 0
        for i in range(model.len):
            ccost += model.layers[i].sumspike.sum()
        ccost_scores.append(ccost)
        acc = 100.0 * correct / total
        print(test_or_train + ' correct: %d accuracy: %.2f%% ccost: %d' % (correct, acc, ccost.data))

    loss_val_8times.append(loss_val)
    plt.plot(loss_val)
    np.savetxt("experiment_TL"+ str(j)+'.txt', loss_val)
    torch.save(model, "model_TL" + str(j) + ".pt")
 

'''
    if if_tb:
        writer.add_scalar('acc_' + test_or_train, acc, epoch + 1)
    if if_test:
        test_scores.append(acc)
            #     torch.save(model1, 'model1.pt')
            #     model = torch.load("model3.pt")
    else:
        train_scores.append(acc)

    for i in range(model.len):
        if find:
            layer = model.layers[i]
            np.save(place + task + 'layer' + str(i) + '_a' + str(epoch + 1), layer.a.detach().cpu().numpy())
            np.save(place + task + 'layer' + str(i) + '_b' + str(epoch + 1), layer.b.detach().cpu().numpy())
            np.save(place + task + 'layer' + str(i) + '_c' + str(epoch + 1), layer.c.detach().cpu().numpy())
            np.save(place + task + 'layer' + str(i) + '_d' + str(epoch + 1), layer.d.detach().cpu().numpy())



    parm1={}
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        parm1[name]=parameters.detach().numpy()
'''


'''
    if (epoch + 1) % 1 == 0:
        eval(epoch, if_test = True)
    if (epoch + 1) % 20 == 0:
        eval(epoch, if_test = False)
    if (epoch + 1) % 20 == 0:
        print('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
        avg = (test_scores[-1] + test_scores[-2] + test_scores[-3] + test_scores[-4] + test_scores[-5] + test_scores[-6] + test_scores[-7] + test_scores[-8] + test_scores[-9] + test_scores[-10]) / 10
        print('Average of Last Ten Test Accuracy : %.2f%%' % (avg))



    parm1={}
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        parm1[name]=parameters.detach().numpy()
        
        
        
A1=parm1['layers.0.a'][:,:]
B1=parm1['layers.0.b'][:,:]
C1=parm1['layers.0.c'][:,:]
D1=parm1['layers.0.d'][:,:]
E1=parm1['layers.0.fc.weight'][:,:]
F1=parm1['layers.0.fc.bias'][:]
G1=parm1['layers.1.a'][:,:]
H1=parm1['layers.1.b'][:,:]
I1=parm1['layers.1.c'][:,:]
J1=parm1['layers.1.d'][:,:]
K1=parm1['layers.1.fc.weight'][:,:]
L1=parm1['layers.1.fc.bias'][:]

AA1=parm1['layers.0.a'][:,:]
AB1=parm1['layers.0.b'][:,:]
AC1=parm1['layers.0.c'][:,:]
AD1=parm1['layers.0.d'][:,:]
AE1=parm1['layers.0.fc.weight'][:,:]
AF1=parm1['layers.0.fc.bias'][:]
AG1=parm1['layers.1.a'][:,:]
AH1=parm1['layers.1.b'][:,:]
AI1=parm1['layers.1.c'][:,:]
AJ1=parm1['layers.1.d'][:,:]
AK1=parm1['layers.1.fc.weight'][:,:]
AL1=parm1['layers.1.fc.bias'][:]

A=(A1+AA1)/2
B=(B1+AB1)/2
C=(C1+AC1)/2
D=(D1+AD1)/2
E=(E1+AE1)/2
F=(F1+AF1)/2
G=(G1+AG1)/2
H=(H1+AH1)/2
I=(I1+AI1)/2
J=(J1+AJ1)/2
K=(K1+AK1)/2
L=(L1+AL1)/2





with torch.no_grad():
    for name, param in model.named_parameters():
        if 'layers.0.a' in name:
            param.copy_(torch.from_numpy(A))
        if 'layers.0.b' in name:
            param.copy_(torch.from_numpy(B))
        if 'layers.0.c' in name:
            param.copy_(torch.from_numpy(C))
        if 'layers.0.d' in name:
            param.copy_(torch.from_numpy(D))
        if 'layers.0.fc.weight' in name:
            param.copy_(torch.from_numpy(E))
        if 'layers.0.fc.bias' in name:
            param.copy_(torch.from_numpy(F))
        if 'layers.1.a' in name:
            param.copy_(torch.from_numpy(G))
        if 'layers.1.b' in name:
            param.copy_(torch.from_numpy(H))
        if 'layers.1.c' in name:
            param.copy_(torch.from_numpy(I))
        if 'layers.1.d' in name:
            param.copy_(torch.from_numpy(J))
        if 'layers.1.fc.weight' in name:
            param.copy_(torch.from_numpy(K))
        if 'layers.1.fc.bias' in name:
            param.copy_(torch.from_numpy(L))
            
'''

'''
A=[ 0.0845, -1.0571,  0.1777, -0.3663, -0.9113,  0.7898,  1.4889,  1.0135,0.9143, -0.2273]
a = np.array(A)
A=torch.from_numpy(A1+10)
'''
