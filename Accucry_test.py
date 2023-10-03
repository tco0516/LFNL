import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from SNN import SNN
import time
import os
import argparse
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import pandas as pd



parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-gpu', type = int, default = 0)
parser.add_argument('-seed', type = int, default = 3154)
parser.add_argument('-nps', type = str, default = 'C1')
parser.add_argument('-dts', type = str, default = 'C10')

opt = parser.parse_args()

#torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
#torch.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True


find = False

test_scores = []
train_scores = []
ccost_scores = []
if_tb = False


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


confusionall=[[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0] , [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]  ]
acc_val_8times=list() 
experiments=5 # 实验次数
for j in range(experiments):
 
    #    model_initial=model
    #     torch.save(model_initial, 'model_initial.pt')
    # model = torch.load("model_initial.pt")
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



  
    model = torch.load("model_TL20.pt")
#    model = torch.load("device3/experiment_UPDATE (" + str(j+1) +")"+ ".pt")
#    model = torch.load("LFNL/experiment_FL_TL" + str(j) + ".pt")
    
    
    
    loss_val=list()

    start_time = time.time()
    total_loss = 0
    
    index = [i for i in range(test_data.shape[0])]
    np.random.shuffle(index) # 打乱索引
    test_data = test_data[index]
    test_data=torch.Tensor(test_data)
    labels_test = labels_test[index]

    
    if_test=1
    model.eval()
    correct = 0
    total = 0
    loss1=list()
    predall=list()
    labels_true=list()
    if task == 'MNIST':
        for i in range (1, int(len(test_data)/hyperparams[0])):
            images=test_data[(i-1)*hyperparams[0]:i*hyperparams[0],:]
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
            predall=np.hstack((predall,pred)) 
            correct += (pred == labels).sum()
            labels_true=np.hstack((labels_true,labels)) 
        correct = correct.item()

    ccost = 0
    for i in range(model.len):
        ccost += model.layers[i].sumspike.sum()
    ccost_scores.append(ccost)
    acc = 100.0 * correct / total
    print('acc',acc)
    acc_val_8times.append(acc)
    confusion1=confusion_matrix(labels_true, predall)
#    confusionall=np.hstack((confusionall,confusion1)) 
    confusionall=confusionall+confusion1
#    confusion.append(confusion1)
AA=np.array(acc_val_8times)
AA.reshape(1,-1)
#confusionall=np.array(confusion)  
#confusionall.reshape(3,-3)
# np.savetxt("acc_val"+ str(j)+'.txt', loss_val)

'''
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



np.save(file="A1.npy", arr=A1)
np.save(file="B1.npy", arr=B1)
np.save(file="C1.npy", arr=C1)
np.save(file="D1.npy", arr=D1)
np.save(file="E1.npy", arr=E1)
np.save(file="F1.npy", arr=F1)
np.save(file="G1.npy", arr=G1)
np.save(file="H1.npy", arr=H1)
np.save(file="I1.npy", arr=I1)
np.save(file="J1.npy", arr=J1)
np.save(file="K1.npy", arr=K1)
np.save(file="L1.npy", arr=L1)
'''