from email.policy import default
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from runx.logx import logx
from yaml import parse
from per import *
from foolbox.distances import l0, l1, l2, linf
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 
from utils import  init_dataloader, init_dataloader1

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success
from collections import defaultdict


import argparse  
import os
from sklearn import cluster

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
import torch.utils.data as Data 
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import lightgbm as lgb
import classifier
from torchvision.models import vgg13,AlexNet
import time
from getdata import *
from modelss import *
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
def attackbyorder(values,groundtruth):
    
    index=values.argsort()
    print(type(index))
    print(index)
    groundtruth=groundtruth[index]
    pred=np.zeros(groundtruth.shape[0])
    pred[:int((groundtruth.shape[0])/2)]=1
    correct=np.sum(pred==groundtruth)
    acc=correct/(groundtruth.shape[0]*1.0)
    # print('correct:',correct)
    # print(groundtruth.shape)
    return acc
	


def clipDataTopX(dataToClip, top=3):
    res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
    return np.array(res)

def getoutput(targetmodel,shadowmodel,data_loader) :
    Toutput=[]
    Soutput=[]
    targets=[]
    targetmodel.eval()
    shadowmodel.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            # Toutput.append(np.array(clipDataTopX(torch.softmax(targetmodel(data),dim=1).cpu(),3)))
            # Soutput.append(np.array(clipDataTopX(torch.softmax(shadowmodel(data),dim=1).cpu(),3)))
            # Toutput.append(np.array(torch.softmax(targetmodel(data),dim=1).cpu()))
            # Soutput.append(np.array(torch.softmax(shadowmodel(data),dim=1).cpu()))
            Toutput.append(np.array(targetmodel(data).cpu()))
            Soutput.append(np.array(shadowmodel(data).cpu()))
            targets.append(int(target.cpu()))
            
    Toutput=np.array(Toutput)
    Toutput=np.squeeze(Toutput)
    Soutput=np.squeeze(Soutput)
    Soutput=np.array(Soutput)

    t_tr_outputs=Toutput[0:int(len(data_loader.dataset)/2),:]
    t_te_outputs=Toutput[int(len(data_loader.dataset)/2):len(data_loader.dataset),:]
    s_tr_outputs=Soutput[0:int(len(data_loader.dataset)/2),:]
    s_te_outputs=Soutput[int(len(data_loader.dataset)/2):len(data_loader.dataset),:]
    t_tr_labels=targets[0:int(len(data_loader.dataset)/2)]
    t_te_labels=targets[int(len(data_loader.dataset)/2):len(data_loader.dataset)]
    s_tr_labels,s_te_labels=t_tr_labels,t_te_labels
   
    return  t_tr_outputs,t_te_outputs,s_tr_outputs,s_te_outputs,t_tr_labels,t_te_labels,s_tr_labels,s_te_labels


def Metric_based_acc(args,metric,targetmodel, shadowmodel, data_loader):
    # num_classes=args.num_classes[args.dataset_ID]
    if metric=='perturbation':
        t_tr_dis,t_te_dis=AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster,  maxitr=50, max_eval=10000)
        s_tr_dis,s_te_dis=AdversaryTwo_HopSkipJump(args, shadowmodel, data_loader, cluster,  maxitr=50, max_eval=10000)
        np.savez(args.root+'/perturbation'+'/'+str(args.cluster)+'/perturbation.npz',t_tr_dis,t_te_dis,s_tr_dis,s_te_dis)
        
        # print(t_tr_dis.shape,t_te_dis.shape,s_tr_dis.shape,s_te_dis.shape)
        targets=[]
        with torch.no_grad():
            for data, target in data_loader:
                targets.append(int(target.cpu()))
        t_tr_labels=targets[0:int(len(data_loader.dataset)/2)]
        t_te_labels=targets[int(len(data_loader.dataset)/2):len(data_loader.dataset)]
        s_tr_labels,s_te_labels=t_tr_labels,t_te_labels
        # print(len(s_tr_labels))
        # print(s_tr_labels)
        metric_attack(args,metric,10,None,None,None,None,t_tr_labels,t_te_labels,s_tr_labels,s_te_labels,t_tr_dis,t_te_dis,s_tr_dis,s_te_dis)
    
    else:
        t_tr_outputs,t_te_outputs,s_tr_outputs,s_te_outputs,t_tr_labels,t_te_labels,s_tr_labels,s_te_labels=getoutput(targetmodel,shadowmodel,data_loader)
        metric_attack(args,metric,10,t_tr_outputs,t_te_outputs,s_tr_outputs,s_te_outputs,t_tr_labels,t_te_labels,s_tr_labels,s_te_labels,None,None,None,None)



def metric_attack(args,metric,num_classes,t_tr_outputs,t_te_outputs,s_tr_outputs,s_te_outputs,t_tr_labels,t_te_labels,s_tr_labels,s_te_labels,t_tr_dis,t_te_dis,s_tr_dis,s_te_dis):
    def log_value( probs, small_value=1e-1):
        return -np.log(np.maximum(probs, small_value))
    

    def entr_comp( probs):

        entropy=np.sum(np.multiply(probs, log_value(probs)),axis=1)
        # print('entropy:',entropy.shape) (2500,)
        # print(np.any(np.isnan(entropy))) False
        return entropy
    
    def m_entr_comp(probs, true_labels):
        log_probs = log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(len(true_labels)), true_labels] = reverse_probs[range(len(true_labels)), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(len(true_labels)), true_labels] = log_probs[range(len(true_labels)), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)

    def loss(probs,true_labels):
        # print(probs.shape)  (2500,10)
        # print(len(true_labels))
        couple=zip(probs,true_labels)
        loss=[F.cross_entropy(torch.from_numpy(prob), torch.tensor(target)) for (prob,target) in couple]
        return np.array(loss)



    def thre_setting( tr_values, te_values): 
        value_list = np.concatenate((tr_values, te_values))   #pick the treshold from the existing values
        thre, max_acc = 0, 0
        # print('train value:',tr_values)
        # print("test value:",te_values)
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        print("picked threshold: {},train acc on shadow model:{}".format(thre,max_acc))
        return thre
        
    def correctness_based():
        t_tr_corr = (np.argmax(t_tr_outputs, axis=1)==t_tr_labels).astype(int)
        t_te_corr = (np.argmax(t_te_outputs, axis=1)==t_te_labels).astype(int)
        t_tr_acc = np.sum(t_tr_corr)/(len(t_tr_corr)+0.0)
        t_te_acc = np.sum(t_te_corr)/(len(t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
    
    def confidence_based():
        s_tr_conf = np.array([s_tr_outputs[i, s_tr_labels[i]] for i in range(len(s_tr_labels))])
        s_te_conf = np.array([s_te_outputs[i, s_te_labels[i]] for i in range(len(s_te_labels))])
        t_tr_conf = np.array([t_tr_outputs[i, t_tr_labels[i]] for i in range(len(t_tr_labels))])
        t_te_conf = np.array([t_te_outputs[i, t_te_labels[i]] for i in range(len(t_te_labels))])
        t_tr_mem, t_te_non_mem = 0, 0
        truth1=np.ones(t_tr_conf.shape[0])
        truth2=np.zeros(t_te_conf.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        # print(groundtruth.shape) (5000,)
        targetvalues=np.concatenate((t_tr_conf,t_te_conf))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=1, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)
        # print(targetvalues.shape)  (5000,)
        acc=attackbyorder(targetvalues,groundtruth[::-1])  
        # for num in range(num_classes):
        #     print("class:",num)
        #     thre = thre_setting(s_tr_conf[np.where(np.array(s_tr_labels)==num)[0]], s_te_conf[np.where(np.array(s_te_labels)==num)[0]])#
        #     t_tr_mem += np.sum(t_tr_conf[np.where(np.array(t_tr_labels)==num)[0]]>=thre) 
        #     t_te_non_mem += np.sum(t_te_conf[np.where(np.array(t_te_labels)==num)[0]]<thre)
        #     print(thre,t_tr_mem,t_te_non_mem)
        # mem_inf_acc = 0.5*(t_tr_mem/(len(t_tr_labels)+0.0) + t_te_non_mem/(len(t_te_labels)+0.0))
        print('confidence-based MIA, the attack acc is {acc:.3f}'.format(acc=acc))
        return
    
    
    def entropy_based():
        # s_tr_entr = entr_comp(s_tr_outputs)
        # s_te_entr = entr_comp(s_te_outputs)
        # t_tr_entr = entr_comp(t_tr_outputs)
        # t_te_entr = entr_comp(t_te_outputs)
        s_tr_entr = entr_comp(clipDataTopX(s_tr_outputs,args.clip))
        s_te_entr = entr_comp(clipDataTopX(s_te_outputs,args.clip))
        t_tr_entr = entr_comp(clipDataTopX(t_tr_outputs,args.clip))
        t_te_entr = entr_comp(clipDataTopX(t_te_outputs,args.clip))
        t_tr_mem, t_te_non_mem = 0, 0
        truth1=np.ones(t_tr_entr.shape[0])
        truth2=np.zeros(t_te_entr.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        # print(groundtruth.shape)   (5000,)
        targetvalues=np.concatenate((t_tr_entr,t_te_entr))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=0, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)
        # print(targetvalues[:100])
        # print(targetvalues[4900:])

        #attack based on order
        acc=attackbyorder(targetvalues,groundtruth)




        #attack based on threshold
        # for num in range(num_classes):
        #     print("class:",num)
        #     thre = thre_setting(s_tr_entr[np.where(np.array(s_tr_labels)==num)[0]], s_te_entr[np.where(np.array(s_te_labels)==num)[0]])
        #     t_tr_mem += np.sum(t_tr_entr[np.where(np.array(t_tr_labels)==num)[0]]>=thre) 
        #     t_te_non_mem += np.sum(t_te_entr[np.where(np.array(t_te_labels)==num)[0]]<thre)

        # mem_inf_acc = 0.5*(t_tr_mem/(len(t_tr_labels)+0.0) + t_te_non_mem/(len(t_te_labels)+0.0))
        # print("decided members number {},decided nonmembers number{}".format(t_tr_mem,t_te_non_mem))
        print('entropy-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return       
    def modientropy_based():
        s_tr_entr = m_entr_comp(s_tr_outputs,s_tr_labels)
        s_te_entr = m_entr_comp(s_te_outputs,s_te_labels)
        t_tr_entr = m_entr_comp(t_tr_outputs,t_tr_labels)
        t_te_entr = m_entr_comp(t_te_outputs,t_te_labels)
        t_tr_mem, t_te_non_mem = 0, 0
        truth1=np.ones(t_tr_entr.shape[0])
        truth2=np.zeros(t_te_entr.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        # print(groundtruth.shape)   (5000,)
        targetvalues=np.concatenate((t_tr_entr,t_te_entr))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=0, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)
        # print(targetvalues[:100])
        # print(targetvalues[4900:])

        #attack based on order
        acc=attackbyorder(targetvalues,groundtruth)

        #attack based on threshold
        # for num in range(num_classes):
        #     print("class:",num)
        #     thre = thre_setting(s_tr_entr[np.where(np.array(s_tr_labels)==num)[0]], s_te_entr[np.where(np.array(s_te_labels)==num)[0]])
        #     t_tr_mem += np.sum(t_tr_entr[np.where(np.array(t_tr_labels)==num)[0]]>=thre) 
        #     t_te_non_mem += np.sum(t_te_entr[np.where(np.array(t_te_labels)==num)[0]]<thre)
        #     print(thre,t_tr_mem,t_te_non_mem)
        # mem_inf_acc = 0.5*(t_tr_mem/(len(t_tr_labels)+0.0) + t_te_non_mem/(len(t_te_labels)+0.0))
        print('modified entropy-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return 
    def loss_based():
        s_tr_entr = loss(s_tr_outputs,s_tr_labels)
        s_te_entr = loss(s_te_outputs,s_te_labels)
        t_tr_entr = loss(t_tr_outputs,t_tr_labels)
        t_te_entr = loss(t_te_outputs,t_te_labels)
        t_tr_mem, t_te_non_mem = 0, 0
        truth1=np.ones(t_tr_entr.shape[0])
        truth2=np.zeros(t_te_entr.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        print(groundtruth.shape)
        targetvalues=np.concatenate((t_tr_entr,t_te_entr))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=0, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)

        #attack based on order
        acc=attackbyorder(targetvalues,groundtruth)
        
        #attack based on threshold
        # for num in range(num_classes):
        #     print("class:",num)
        #     thre = thre_setting(s_tr_entr[np.where(np.array(s_tr_labels)==num)[0]], s_te_entr[np.where(np.array(s_te_labels)==num)[0]])
        #     t_tr_mem += np.sum(t_tr_entr[np.where(np.array(t_tr_labels)==num)[0]]>=thre) 
        #     t_te_non_mem += np.sum(t_te_entr[np.where(np.array(t_te_labels)==num)[0]]<thre)
        #     print(thre,t_tr_mem,t_te_non_mem)
        mem_inf_acc = 0.5*(t_tr_mem/(len(t_tr_labels)+0.0) + t_te_non_mem/(len(t_te_labels)+0.0))
        print('loss-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return 
    # print('training size:',args.cluster)

    def per_based():
        s_tr_per=s_tr_dis
        s_te_per=s_te_dis
        t_tr_per=t_tr_dis
        t_te_per=t_te_dis
        t_tr_mem, t_te_non_mem = 0, 0

        truth1=np.ones(t_tr_per.shape[0])
        truth2=np.zeros(t_te_per.shape[0])
        groundtruth=np.concatenate((truth1,truth2))

        targetvalues=np.concatenate((t_tr_per,t_te_per))
        # print("target values shape:",targetvalues.shape)# 
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=1, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)

        #attack based on order
        acc=attackbyorder(np.squeeze(targetvalues),groundtruth[::-1])
        # for num in range(num_classes):
        #     print("class:",num)
        #     thre = thre_setting(s_tr_per[np.where(np.array(s_tr_labels)==num)[0]], s_te_per[np.where(np.array(s_te_labels)==num)[0]])
        #     t_tr_mem += np.sum(t_tr_per[np.where(np.array(t_tr_labels)==num)[0]]>=thre)
        #     t_te_non_mem += np.sum(t_te_per[np.where(np.array(t_te_labels)==num)[0]]<thre)
        #     print(thre,t_tr_mem,t_te_non_mem)
        # mem_inf_acc = 0.5*(t_tr_mem/(len(t_tr_labels)+0.0) + t_te_non_mem/(len(t_te_labels)+0.0))
        print('perturbation-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return 
    print('training size:',args.cluster)

    if metric=='correctness': correctness_based()
    elif metric=='confidence': confidence_based()
    elif metric=='entropy':    entropy_based()
    elif metric=='modifiedentropy': modientropy_based()
    elif metric=='loss': loss_based()
    elif metric=='perturbation':per_based()
    

def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def get_dataset(args):
    cluster=args.cluster
    if args.split_type=='torch':
        trainset=torchvision.datasets.CIFAR10('../datasets', train=True, download=True,
                            transform=transforms.Compose(
                                [
                                    #  transforms.RandomCrop(32, padding=4),
                                    #  transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))

        testset=torchvision.datasets.CIFAR10('../datasets', train=False, download=True, 
                            transform=transforms.Compose(
                                [
                                    #  transforms.RandomCrop(32, padding=4),
                                    #  transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))

        totalset=Data.ConcatDataset([trainset,testset])
        trainT,testT,trainS,testS=Data.random_split(dataset=totalset,lengths=[cluster,cluster,int((60000-2*cluster)/2),int((60000-2*cluster)/2)])

    elif args.split_type=='lasagne':
        if (args.preprocess):
            root='/datasets'
            targetTrain, targetTrainLabel, targetTest, targetTestLabel, shadowTrain, shadowTrainLabel, shadowTest, shadowTestLabel=initializeData(args,root,args.num_shadow,dataFolderPath = './data/')            
            shadowTrain, shadowTrainLabel=np.squeeze(shadowTrain),np.squeeze(shadowTrainLabel)
            shadowTest, shadowTestLabel=np.squeeze(shadowTest),np.squeeze(shadowTestLabel)

        else: #load the preprocessed data
            if args.defend ==1 :
                targetTrain, targetTrainLabel  = load_data('/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) +'/targetTrain.npz')
                targetTest,  targetTestLabel   = load_data('/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) + '/targetTest.npz')
                shadowTrain, shadowTrainLabel  = load_data('/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) + '/shadowTrain.npz')
                shadowTest,  shadowTestLabel   = load_data('/'+args.dataset+'/Preprocessed'+'/'+str(args.cluster) + '/shadowTest.npz')
                shadowTrain, shadowTrainLabel=np.squeeze(shadowTrain),np.squeeze(shadowTrainLabel)
                shadowTest, shadowTestLabel=np.squeeze(shadowTest),np.squeeze(shadowTestLabel)
            else: 
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


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    description = "loss={:.4f} acc={:.2f}%"
    total = 0
    # with tqdm(train_loader) as batch:
        # for idx, (data, target) in enumerate(batch):
    for idx, (data, target) in enumerate(train_loader):
        
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()
            output = model(data)
            target=target.to(torch.int64)
            loss = F.cross_entropy(output, target)
            # loss =  pt_categorical_crossentropy(output,target)

            total += data.shape[0]
            total_loss += loss.detach().item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # batch.set_description(description.format(total_loss / total, 100*correct / total))

            loss.backward()
            optimizer.step()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            target=target.to(torch.int64)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            # test_loss +=  pt_categorical_crossentropy(output,target)
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    # print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
    #       .format(test_loss, acc))
    return acc, test_loss

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def clipDataTopX(dataToClip, top=3):
    res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
    return np.array(res)





def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--schedule',default=[25,40,50,75],help='when to change lr')
    parser.add_argument('--num_shadow',type=int,default=1)
    parser.add_argument('--cluster',type=int,default=2500)
    parser.add_argument('--framework',type=str,default='torch',help='the attack model framework')
    parser.add_argument('--train', default=0, action='store_true',help='whether to train the target and shadow models')
    parser.add_argument('--weightdecay',type=float,default=5e-4,help='l2 regularization')
    parser.add_argument('--split_type',type=str,default='lasagne',help="how to prepare the target and shadow training data")
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data, if false then load preprocessed data')
    parser.add_argument('--dataset',default='celeba')
    parser.add_argument('--metric',default='confidence',help='entropy,correctness,confidence,modifiedentropy,loss')
    parser.add_argument('--dis',default='l0',help="perturbation type")
    parser.add_argument('--root',type=str,default='./result_metric')
    parser.add_argument('--clip',type=int,default=3)
    parser.add_argument('--classes',type=int,default=10)
    parser.add_argument('--defend',type=int,default=1)
    parser.add_argument('--lamda',type=str,default=1.0)
    parser.add_argument('--classnumber',type=str,default='1000')
    parser.add_argument('--trainingsize',type=str,default='10')

    

    args = parser.parse_args()
    root='./result_metric'+'/'+args.metric
    args.root=root
    datapath=root+'/data'
    modelpath=root+'/model'
    cluster=args.cluster
    
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
    else: 
        trainT, testT, trainS, testS=get_dataset(args)
    attackdataset=Data.ConcatDataset([trainT,testT])  
    thismodelpath=modelpath+'/'+str(cluster)
    perturbationpath=root+'/perturbation'+'/'+str(args.cluster)
    try:
        os.makedirs(thismodelpath)
    except OSError:
        pass
    try:
        os.makedirs(perturbationpath)
    except OSError:
        pass

    if args.train:
        #train target
        # model = CNNCifar10().cuda()
        if args.dataset=='cifar10':
            model=Net(args.classes).cuda()
        elif args.dataset=='MNIST':
            model=Net1().cuda()
        elif args.dataset=='celeba':
            model=VGG16(int(args.classnumber)).cuda()
        else:#if args.dataset=='tinyImageNet':
            print('exp on tinyimagenet')
            # model=Net2().cuda()
            # model=vgg13().cuda()
            model=AlexNet(num_classes=43).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)#,weight_decay=args.weightdecay)
        best_acc = -1
        train_loaderT = Data.DataLoader(trainT,
            batch_size=256, shuffle=True, num_workers=4)
        test_loaderT = Data.DataLoader(testT,
            batch_size=256, shuffle=True, num_workers=4)
        # for epoch in tqdm(range(1, args.epochs + 1)):
        for epoch in range(1, args.epochs + 1):

            # print('epoch:',epoch)
            adjust_learning_rate(optimizer, epoch, args)
            train(model, train_loaderT, optimizer)
            acc, loss = test(model, test_loaderT)
            if acc > best_acc:
                torch.save(model.state_dict(), thismodelpath+'/targetmodel.pt')
            best_acc = max(best_acc, acc)
        attack_x, attack_y = [], []
        # model = CNNCifar10().cuda()
        
        # if args.dataset=='cifar10':
        #     model= Net(args.classes).cuda()
        # elif args.dataset=='MNIST':
        #     model=Net1().cuda()
        # else:#if args.dataset=='tinyImageNet':
        #     # model=Net2().cuda()
        #     # model=vgg13().cuda()
        #     model=AlexNet(num_classes=43).cuda()

        

        
        print("Finish training target model")


        #train shadow
        for i in range(args.num_shadow):
            if args.split_type=='torch':
                trains, res=Data.random_split(trainS,lengths=[cluster,int((60000-2*cluster)/2-cluster)])
                tests,  res=Data.random_split(testS,lengths=[cluster,int((60000-2*cluster)/2-cluster)])
            elif args.split_type=='lasagne':
                trains,tests=trainS,testS
            train_loaderS = Data.DataLoader(trains,
            batch_size=256, shuffle=True, num_workers=4)
            test_loaderS = torch.utils.data.DataLoader(tests,
            batch_size=256, shuffle=True, num_workers=4)
            # model = CNNCifar10().cuda()
            if args.dataset=='cifar10':
                model= Net(args.classes).cuda()
            elif args.dataset=='MNIST':
                model=Net1().cuda()
            elif args.dataset=='celeba':
                model=VGG16(int(args.classnumber)).cuda()
            else:#if args.dataset=='tinyImageNet':
                # model=Net2().cuda()
                # model=vgg13().cuda()
                model=AlexNet(num_classes=43).cuda()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)#,weight_decay=args.weightdecay)
            best_acc = -1

            # for epoch in tqdm(range(1, args.epochs + 1)):
            for epoch in range(1, args.epochs + 1):

                # print('epoch:',epoch)
                adjust_learning_rate(optimizer, epoch, args)
                train(model, train_loaderS, optimizer)
                acc, loss = test(model, test_loaderS)
                if acc > best_acc:
                    torch.save(model.state_dict(), thismodelpath+'/number'+str(i)+'model.pt')
                best_acc = max(best_acc, acc)
            
            
            

        print("Finish training all shadow models")

    else: pass 

    # metric attack
    
    # load target and shadow model

    if args.dataset=='cifar10':
        targetmodel= Net(args.classes).cuda()
        shadowmodel= Net(args.classes).cuda()
    elif args.dataset=='MNIST':
        targetmodel=Net1().cuda()
        shadowmodel=Net1().cuda()
    elif args.dataset == 'celeba':
        targetmodel = VGG16(int(args.classnumber)).cuda()
        shadowmodel = VGG16(int(args.classnumber)).cuda()


    else:#if args.dataset=='tinyImageNet':
    # model=Net2().cuda()
    # model=vgg13().cuda()
        targetmodel=AlexNet(num_classes=43).cuda()
        shadowmodel=AlexNet(num_classes=43).cuda()
    if args.defend == 1:
        print('load defended model:')
        checkpoint=torch.load('/results/checkpoints_celeba/epoch49alpha' + args.lamda +'attackoriattack')
        targetmodel.load_state_dict(checkpoint['state_dict'])
    else:
        print('load normal model:')
        targetmodel.load_state_dict(torch.load(thismodelpath+'/targetmodel.pt'))
    


    # shadowmodel.load_state_dict(torch.load(thismodelpath+'/number'+str(0)+'model.pt'))
    
    attackdataloader=Data.DataLoader(attackdataset,batch_size=1, shuffle=False, num_workers=4)
    Metric_based_acc(args,args.metric,targetmodel, shadowmodel, attackdataloader)
  



if __name__ == '__main__':

            main()