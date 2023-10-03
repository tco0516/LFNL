import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from SNN import SNN
import time
import os
import shutil
import argparse
from tensorboardX import SummaryWriter
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
    #model.cuda()
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
    
    optimizer1 = torch.optim.Adam(paras_new)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size = 15, gamma = 0.9)
    
    optimizer2 = torch.optim.Adam(paras_new)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size = 15, gamma = 0.9)
    
    optimizer3 = torch.optim.Adam(paras_new)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size = 15, gamma = 0.9)
    
    cossim = torch.nn.CosineSimilarity(dim = 1, eps = 1e-8)
    sigmoid = torch.nn.Sigmoid()
    norm = torch.nn.BatchNorm2d(1)
    
    
    
    
    #        model3=model
      #               torch.save(model3, 'model3.pt')
      #               model = torch.load("model2.pt")
    
    #model = torch.load("model1.pt")
    
    model1=model
    model2=model
    model3=model
    loss_val=list()
    for epoch in range(num_epoch):
        model1.train()
    
        scheduler1.step()
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
                    optimizer1.zero_grad()
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
    
                    outputs = model1(images)
                    loss = loss_function(outputs, labels)
                    total_loss += float(loss)
                    loss.backward(retain_graph = True)
                    optimizer1.step()
    #                print('Train loss',float(loss))
                    
                    '''
                    #Valiation
                    total_loss1 = 0
                    for i in range (1, int(len(test_data)/hyperparams[0])):
                        images1=test_data[(i-1)*hyperparams[0]:i*hyperparams[0],:,:]
                        labels1=labels_test[(i-1)*hyperparams[0]:i*hyperparams[0]]
                        one_hot1 = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels1.unsqueeze(1), 1)
                        labels1 = Variable(one_hot1)
                        outputs1 = model(images1)
                        loss1 = loss_function(outputs1, labels1)
                        total_loss1 += float(loss1)
                    print('Val loss',total_loss1/i)
                    '''
#        torch.save(model1, 'model1.pt')
        
        
        
        scheduler2.step()
        model2.train()
        for i in range (1, int(len(imag2)/hyperparams[0])):
            if True:
                images=imag2[(i-1)*hyperparams[0]:i*hyperparams[0],:,:,:]
                labels=labe2[(i-1)*hyperparams[0]:i*hyperparams[0]]
            # if i < 60:
                if 1 == 1:
                    optimizer2.zero_grad()
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
    
                    outputs = model2(images)
                    loss = loss_function(outputs, labels)
                    total_loss += float(loss)
                    loss.backward(retain_graph = True)
                    optimizer2.step()
    #                print('loss',loss)
#        torch.save(model2, 'model2.pt')
    
        scheduler3.step()
        model3.train()
    #    if epoch==1:
    #        model3=model1
        for i in range (1, int(len(imag3)/hyperparams[0])):
            if True:
                images=imag3[(i-1)*hyperparams[0]:i*hyperparams[0],:,:,:]
                labels=labe3[(i-1)*hyperparams[0]:i*hyperparams[0]]
            # if i < 60:
                if 1 == 1:
                    optimizer3.zero_grad()
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
    
                    outputs = model3(images)
                    loss = loss_function(outputs, labels)
                    total_loss += float(loss)
                    loss.backward(retain_graph = True)
                    optimizer3.step()
    #                print('loss',loss)
#        torch.save(model3, 'model3.pt')
    
    
    
#        model1 = torch.load("model1.pt")
        parm1={}
        for name,parameters in model1.named_parameters():
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
    
#        model2 = torch.load("model2.pt")
        parm2={}
        for name,parameters in model2.named_parameters():
            parm2[name]=parameters.detach().numpy()
        AA1=parm2['layers.0.a'][:,:]
        AB1=parm2['layers.0.b'][:,:]
        AC1=parm2['layers.0.c'][:,:]
        AD1=parm2['layers.0.d'][:,:]
        AE1=parm2['layers.0.fc.weight'][:,:]
        AF1=parm2['layers.0.fc.bias'][:]
        AG1=parm2['layers.1.a'][:,:]
        AH1=parm2['layers.1.b'][:,:]
        AI1=parm2['layers.1.c'][:,:]
        AJ1=parm2['layers.1.d'][:,:]
        AK1=parm2['layers.1.fc.weight'][:,:]
        AL1=parm2['layers.1.fc.bias'][:]
    
#        model3 = torch.load("model3.pt")
        parm3={}
        for name,parameters in model3.named_parameters():
            parm3[name]=parameters.detach().numpy()
        AAA1=parm3['layers.0.a'][:,:]
        AAB1=parm3['layers.0.b'][:,:]
        AAC1=parm3['layers.0.c'][:,:]
        AAD1=parm3['layers.0.d'][:,:]
        AAE1=parm3['layers.0.fc.weight'][:,:]
        AAF1=parm3['layers.0.fc.bias'][:]
        AAG1=parm3['layers.1.a'][:,:]
        AAH1=parm3['layers.1.b'][:,:]
        AAI1=parm3['layers.1.c'][:,:]
        AAJ1=parm3['layers.1.d'][:,:]
        AAK1=parm3['layers.1.fc.weight'][:,:]
        AAL1=parm3['layers.1.fc.bias'][:]
    
    
    
        A=(A1+AA1+AAA1)/3
        B=(B1+AB1+AAB1)/3
        C=(C1+AC1+AAC1)/3
        D=(D1+AD1+AAD1)/3
        E=(E1+AE1+AAE1)/3
        F=(F1+AF1+AAF1)/3
        G=(G1+AG1+AAG1)/3
        H=(H1+AH1+AAH1)/3
        I=(I1+AI1+AAI1)/3
        J=(J1+AJ1+AAJ1)/3
        K=(K1+AK1+AAK1)/3
        L=(L1+AL1+AAL1)/3
    
    
    
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
                
        model1=model
        model2=model
        model3=model
        
        
    
                         
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
    np.savetxt("experiment_LFL"+ str(j)+'.txt', loss_val)
    torch.save(model, "model_LFL" + str(j) + ".pt")


#    torch.save(model1, "experiment_" + str(j) + ".pt")


'''
    if if_tb:
        writer.add_scalar('acc_' + test_or_train, acc, epoch + 1)
    if if_test:
        test_scores.append(acc)
            #     torch.save(model, 'model.pt')
            #     model = torch.load("model.pt")
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


model1 = torch.load("model1.pt")
parm1={}
for name,parameters in model1.named_parameters():
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


model2 = torch.load("model2.pt")
parm2={}
for name,parameters in model2.named_parameters():
    print(name,':',parameters.size())
    parm2[name]=parameters.detach().numpy()
AA1=parm2['layers.0.a'][:,:]
AB1=parm2['layers.0.b'][:,:]
AC1=parm2['layers.0.c'][:,:]
AD1=parm2['layers.0.d'][:,:]
AE1=parm2['layers.0.fc.weight'][:,:]
AF1=parm2['layers.0.fc.bias'][:]
AG1=parm2['layers.1.a'][:,:]
AH1=parm2['layers.1.b'][:,:]
AI1=parm2['layers.1.c'][:,:]
AJ1=parm2['layers.1.d'][:,:]
AK1=parm2['layers.1.fc.weight'][:,:]
AL1=parm2['layers.1.fc.bias'][:]


model3 = torch.load("model3.pt")
parm3={}
for name,parameters in model3.named_parameters():
    print(name,':',parameters.size())
    parm3[name]=parameters.detach().numpy()
AAA1=parm3['layers.0.a'][:,:]
AAB1=parm3['layers.0.b'][:,:]
AAC1=parm3['layers.0.c'][:,:]
AAD1=parm3['layers.0.d'][:,:]
AAE1=parm3['layers.0.fc.weight'][:,:]
AAF1=parm3['layers.0.fc.bias'][:]
AAG1=parm3['layers.1.a'][:,:]
AAH1=parm3['layers.1.b'][:,:]
AAI1=parm3['layers.1.c'][:,:]
AAJ1=parm3['layers.1.d'][:,:]
AAK1=parm3['layers.1.fc.weight'][:,:]
AAL1=parm3['layers.1.fc.bias'][:]



A=(A1+AA1+AAA1)/3
B=(B1+AB1+AAB1)/3
C=(C1+AC1+AAC1)/3
D=(D1+AD1+AAD1)/3
E=(E1+AE1+AAE1)/3
F=(F1+AF1+AAF1)/3
G=(G1+AG1+AAG1)/3
H=(H1+AH1+AAH1)/3
I=(I1+AI1+AAI1)/3
J=(J1+AJ1+AAJ1)/3
K=(K1+AK1+AAK1)/3
L=(L1+AL1+AAL1)/3



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
