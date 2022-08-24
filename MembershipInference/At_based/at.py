
# coding: utf-8

# In[23]:


from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch.utils.data as Data 
from getdata import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np
from utils1 import init_dataloader, init_dataloader1

# In[2]:


use_cuda = torch.cuda.is_available()


# In[3]:


use_cuda


# In[4]:



manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

best_acc = 0  # best test accuracy


# In[25]:


#Celeba dataset
class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        
        return res


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Net(nn.Module):  
    def __init__(self,classes):
        super(Net,self).__init__()
        self.query_num = 0
        self.conv1=nn.Conv2d(3,32,5)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,32,5)
        self.pool2=nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, classes)
    def forward(self, x):
        self.query_num +=1
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        # x = torch.softmax(self.fc2(x),dim=1)
        x = self.fc2(x)
        # x = self.fc2(x)
        return x

def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model


# In[26]:


class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        print('attack model num classes:',num_classes)
        super(InferenceAttack_HZ, self).__init__()
      
        self.features=nn.Sequential(
            nn.Linear(num_classes,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*2,256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            # print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
                # print (key)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,x,l):
        # print(x.shape)  torch.size([100,10])
        out_x = self.features(x)
        out_l = self.labels(l)
        
        
        is_member =self.combine( torch.cat((out_x  ,out_l),1))
        
        
        return self.output(is_member)



class attack(nn.Module): 
    def __init__(self,clip=3):
        super(attack, self).__init__()

        self.fc1 = nn.Linear(clip , 64)
        # self.fc2 = nn.Linear(64, 100)
        # self.fc3 = nn.Linear(500,100)
        # self.drop = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.drop(x)
        x = self.fc4(x)
        return x  


# In[27]:



train_batch=100
test_batch=100
lr=0.05
epochs=50
state={}
state['lr']=lr


# In[52]:



def train_privatly(args,num_classes,trainloader, model,inference_model, criterion, optimizer, epoch, use_cuda,num_batchs=10000,alpha=0.9):
    # switch to train mode
    #train target model with regularization
    model.train()
    inference_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    

    first_id = -1
    for batch_idx, (inputs, targets) in (trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if first_id == -1:
            first_id = batch_idx
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        
        one_hot_tr = torch.from_numpy((np.zeros((outputs.size(0),num_classes))-1)).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)   #将label转为one-hot vector

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        # print('infer input one hot shape:',infer_input_one_hot.shape)  torch.size([100,10])
        # print('outputs shape:',outputs.shape)   torch.size([100,10])
        if args.attack == 'oriattack':
        
            inference_output = inference_model ( outputs,infer_input_one_hot)  #attack model gain
            #print (inference_output.mean())
            

        elif args.attack == 'selfattack':
            inference_output0 = inference_model(outputs)
            inference_sigmoid = torch.softmax(inference_output0,dim=1)
            inference_output = [ 1-i[1] for i in inference_sigmoid]
            # print(torch.stack(inference_output).shape)  torch.size([100])
            inference_output=torch.stack(inference_output)
            # inference_output = torch.from_numpy(inference_output)
        loss = criterion(outputs, targets) + (alpha)*(((inference_output-1.0).pow(2).mean())) #attack model gain regularization

        # measure accuracy and record loss
        print('check.........')
        print(outputs.data.shape)
        print(targets.shape)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx%5==0:

            print  ('train target model with regularization:','({batch}/{size})  | Loss: {loss:.4f} | top1 acc: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx ,
                    size=args.cluster/train_batch,  #batchsize=100 一共500个batch
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))
        if batch_idx-first_id >= num_batchs:
            break

    return (losses.avg, top1.avg)  #target model acc


# In[44]:



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    #train target model normally
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx%100==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg)


# In[45]:


def test(testloader, model, criterion, epoch, use_cuda):
    #target model test  return top1 acc
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100==0:
            
            print ('test target model:','({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))

    return (losses.avg, top1.avg)


# In[46]:


def privacy_train(args,num_classes,trainloader, model,inference_model, criterion, optimizer, epoch, use_cuda,num_batchs=1000):
    #train attack model
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()
    
    inference_model.train()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    first_id = -1
    for batch_idx,((tr_input, tr_target) ,(te_input, te_target)) in trainloader:
        # measure data loading time
        if first_id == -1:
            first_id = batch_idx
        
        data_time.update(time.time() - end)
        tr_input = tr_input.cuda()
        te_input = te_input.cuda()
        tr_target = tr_target.cuda()
        te_target = te_target.cuda()
        
        
        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)
        
        # compute output
        model_input =torch.cat((v_tr_input,v_te_input)) #按行拼接，就是把train test数据集合并
        
        pred_outputs = model(model_input)
        
        infer_input= torch.cat((v_tr_target,v_te_target))
        
        mtop1, mtop5 =accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        
        mtop1_a.update(mtop1.item(), model_input.size(0))
        mtop5_a.update(mtop5.item(), model_input.size(0))

        
        
        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0),num_classes))-1)).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        

        attack_model_input = pred_outputs#torch.cat((pred_outputs,infer_input_one_hot),1)
        

        if args.attack == 'oriattack':
        
            member_output = inference_model (attack_model_input,infer_input_one_hot)  #attack model gain
            #print (inference_output.mean())
            # print(member_output)
        

        elif args.attack == 'selfattack':
            # print('check')
            inference_output0 = inference_model(attack_model_input)
            inference_sigmoid = torch.softmax(inference_output0,dim=1)
            inference_output = [ i[0] for i in inference_sigmoid]
            # print(torch.stack(inference_output).shape)  torch.size([100])
            member_output=torch.stack(inference_output)
            # print(member_output)
           


        is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_input.size(0)),np.ones(v_te_input.size(0)))),[-1,1])).cuda() #ground-truth membership label
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)
        # print(v_is_member_labels)
        # assert(0)
        # print('membership groundtruth label size:',v_is_member_labels.shape)  torch.size([200,1])
        # print("attack model output size:",member_output.shape)   batch_privacy=100是 torch.size([200,1])

        # print(member_output,v_is_member_labels)
        loss = criterion(member_output, v_is_member_labels) #attack model 的loss  MSEloss 输入 (logits, one-hot)

        # measure accuracy and record loss  of **attack model**
        prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
        losses.update(loss.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0)) 

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx-first_id > num_batchs:
            break

        # plot progress
        if batch_idx%5==0:
            print  ('train attack model:','({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx ,
                    size=args.cluster/train_batch,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))

    return (losses.avg, top1.avg)  #attack model acc


# In[47]:


def privacy_test(trainloader, model,inference_model, criterion, optimizer, epoch, use_cuda,num_batchs=1000):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()
    
    inference_model.eval()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    for batch_idx,((tr_input, tr_target) ,(te_input, te_target)) in trainloader:
        # measure data loading time
        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)
        tr_input = tr_input.cuda()
        te_input = te_input.cuda()
        tr_target = tr_target.cuda()
        te_target = te_target.cuda()
        
        
        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)
        
        # compute output
        model_input =torch.cat((v_tr_input,v_te_input))
        
        pred_outputs = model(model_input)
        
        infer_input= torch.cat((v_tr_target,v_te_target))
        
        mtop1, mtop5 =accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        
        mtop1_a.update(mtop1[0], model_input.size(0))
        mtop5_a.update(mtop5[0], model_input.size(0))

        
        
        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0),100))-1)).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        

        attack_model_input = pred_outputs#torch.cat((pred_outputs,infer_input_one_hot),1)
        member_output = inference_model(attack_model_input,infer_input_one_hot)
        
        
        
        is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_input.size(0)),np.ones(v_te_input.size(0)))),[-1,1])).cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
        losses.update(loss.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx-first_id >= num_batchs:
            break

        # plot progress
#         if batch_idx%10==0:
#             print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
#                     batch=batch_idx ,
#                     size=len(trainloader),
#                     data=data_time.avg,
#                     bt=batch_time.avg,
#                     loss=losses.avg,
#                     top1=top1.avg,
#                     ))

    return (losses.avg, top1.avg)

# In[48]:

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [20,40]:
        state['lr'] *= 0.1 
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


# In[49]:

def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def get_dataset(args):
    cluster=args.cluster
    if (args.preprocess):
        root='/datasets'
        targetTrain, targetTrainLabel, targetTest, targetTestLabel, shadowTrain, shadowTrainLabel, shadowTest, shadowTestLabel=initializeData(args,root,args.num_shadow,dataFolderPath = './data/')            
        shadowTrain, shadowTrainLabel=np.squeeze(shadowTrain),np.squeeze(shadowTrainLabel)
        shadowTest, shadowTestLabel=np.squeeze(shadowTest),np.squeeze(shadowTestLabel)

    else: #load the preprocessed data
        targetTrain, targetTrainLabel  = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) +'/targetTrain.npz')
        targetTest,  targetTestLabel   = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) + '/targetTest.npz')
        shadowTrain, shadowTrainLabel  = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) + '/shadowTrain.npz')
        shadowTest,  shadowTestLabel   = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) + '/shadowTest.npz')
        shadowTrain, shadowTrainLabel=np.squeeze(shadowTrain),np.squeeze(shadowTrainLabel)
        shadowTest, shadowTestLabel=np.squeeze(shadowTest),np.squeeze(shadowTestLabel)

    trainT=Data.TensorDataset(torch.from_numpy(targetTrain),torch.from_numpy(targetTrainLabel))
    testT=Data.TensorDataset(torch.from_numpy(targetTest),torch.from_numpy(targetTestLabel))
    trainS=Data.TensorDataset(torch.from_numpy(shadowTrain),torch.from_numpy(shadowTrainLabel))
    testS=Data.TensorDataset(torch.from_numpy(shadowTest),torch.from_numpy(shadowTestLabel))


    return trainT, testT, trainS, testS

def save_checkpoint_adversary(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_adversary_best.pth.tar'))


# In[50]:

def main():
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=int, default=2500)
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--num_class',type=int,default=1000)
    parser.add_argument('--dataset',type=str,default='celeba')
    parser.add_argument('--num_shadow',type=int,default=1)
    parser.add_argument('--attack',default='oriattack')
    parser.add_argument('--lamda',type=float, default=1)
    parser.add_argument('--trainingsize', type=str, default='10',help='input batch size for training (default: 64)')
    parser.add_argument('--classnumber',type=str,default='1000')
    args = parser.parse_args()



    if args.dataset=='cifar10':
         checkpoint_path='/results/checkpoints_100cifar_alexnetdefense'
    elif args.dataset=='celeba':
         checkpoint_path='/results/checkpoints_celeba'




    if not os.path.isdir(checkpoint_path):
        mkdir_p(checkpoint_path)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        
        
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # prepare test data parts
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
        


    num_classes = args.num_class
    if args.dataset == 'celeba':
        targettrain = '/celeba_split/direct_split/'+args.classnumber+'_'+args.trainingsize+'/target_train.txt'
        targettest = '/celeba_split/direct_split/'+args.classnumber+'_'+args.trainingsize+'/target_test.txt'
        shadowtrain = '/celeba_split/direct_split/'+args.classnumber+'_'+args.trainingsize+'/shadow_train.txt'
        shadowtest = '/celeba_split/direct_split/'+args.classnumber+'_'+args.trainingsize+'/shadow_test.txt'
        trainT = init_dataloader( targettrain, args)   
        testT = init_dataloader( targettest, args)
        trainS = init_dataloader1( shadowtrain, args)
        testS = init_dataloader1( shadowtest, args)

        # dataset_celeba, dataloader_celeba = init_dataloader( filepath, 16, mode="target")
        # print(dir(dataset_celeba))
        # print(dataset_celeba.data.shape)
        # print(dataset_celeba.targets.shape)
        print('successfully load celeba dataset')
    else:    trainT, testT, trainS, testS=get_dataset(args)


    batch_privacy=100
    trainloader = data.DataLoader(trainT, batch_size=batch_privacy, shuffle=True, num_workers=1)
    trainloader_private = data.DataLoader(trainT, batch_size=batch_privacy, shuffle=True, num_workers=1)

    testloader = data.DataLoader(testT, batch_size=batch_privacy, shuffle=True, num_workers=1)

    # Model
    print("==> creating model ")
    # model = AlexNet(num_classes)
    if args.dataset == 'cifar10':
        model= Net(num_classes).cuda()
    elif args.dataset == 'celeba':
        model= VGG16(1000).cuda()
        
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    criterion_attack = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    print('num classes:',num_classes)
    if (args.attack=='oriattack'):
        inference_model = InferenceAttack_HZ(num_classes).cuda()
    elif (args.attack=='selfattack'):
        inference_model = attack(10).cuda()

    elif (args.attack=='trained'):
        inference_model = attack(10).cuda()
        inference_model.load_state_dict(torch.load('/result/model/2500/attackmodel79.02.pt'))
    private_train_criterion = nn.MSELoss()

    optimizer_mem = optim.Adam(inference_model.parameters(), lr=0.00001)


    # In[40]:
    is_best=False
    best_acc=0.0
    start_epoch=0
    # Train and val
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, state['lr']))

        train_enum = enumerate(trainloader)  #50000 

        
        train_private_enum = enumerate(zip(trainloader_private,testloader))  #zip 50000,10000  --> 10000的元组
        for i in range((args.cluster//batch_privacy)//2):  #一个循环里面会train 2*num_batch个batch
            # print('i',i)
            if epoch>3:
                #3轮之后，交替训练target model，attack model  (model,inference model)
                #train attack model
                privacy_loss, privacy_acc = privacy_train(args,num_classes,train_private_enum,model,inference_model,criterion_attack,optimizer_mem,epoch,use_cuda,5) #train attack
                #train target model
                train_loss, train_acc = train_privatly(args,num_classes,train_enum, model,inference_model, criterion, optimizer, epoch, use_cuda,1,args.lamda) #train target
                
                
                if i%10 ==0:
                    print ('privacy res',privacy_acc,train_acc)   #attack acc,  target train acc
                if  (i+1)%(args.cluster/(2*5*batch_privacy)) ==0:
                    train_private_enum = enumerate(zip(trainloader_private,testloader))   #target model一个i只训练2*1个batch  attack model一个i训练2*k个batch所以attackmodel的emu会提前没，要定期更新
            else:
                train_loss, train_acc = train_privatly(args,num_classes,train_enum, model,inference_model, criterion, optimizer, epoch, use_cuda,1000,0)
                break  #note
            
            
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        print ('last test acc',test_acc)


        # save model
        is_best = test_acc>best_acc
        best_acc = max(test_acc, best_acc)
        if epoch == 49 :
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, False, checkpoint=checkpoint_path,filename='epoch{}alpha{}attack{}'.format(epoch,args.lamda,args.attack))


    print('Best target test acc:')
    print(best_acc)

main()