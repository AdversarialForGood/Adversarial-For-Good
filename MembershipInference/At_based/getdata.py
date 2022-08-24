import sys
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.model_selection import train_test_split
import random
import lasagne
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm
import torch.utils.data as Data 
import torch.optim as optim
from torchvision import datasets
import sys
sys.path.append('/home/baiyj/sok')
import dataset
from dataset import load_dataset

def readCIFAR10(data_path):
	for i in range(5):
		f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
		train_data_dict = pickle.load(f,encoding='bytes')
		f.close()
		if i == 0:
			X = train_data_dict[b"data"]
			y = train_data_dict[b"labels"]
			continue
		X = np.concatenate((X , train_data_dict[b"data"]),   axis=0)
		y = np.concatenate((y , train_data_dict[b"labels"]), axis=0)
	f = open(data_path + '/test_batch', 'rb')
	test_data_dict = pickle.load(f,encoding='bytes')
	f.close()
	XTest = np.array(test_data_dict[b"data"])
	yTest = np.array(test_data_dict[b"labels"])
	return X, y, XTest, yTest


def readCIFAR10(data_path):
	totaldata=[]
	totallabel=[]
	for i in range(5):
		f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
		train_data_dict = pickle.load(f,encoding='bytes')
		f.close()
		if i == 0:
			X = train_data_dict[b"data"]
			y = train_data_dict[b"labels"]
			continue
		X = np.concatenate((X , train_data_dict[b"data"]),   axis=0)
		y = np.concatenate((y , train_data_dict[b"labels"]), axis=0)
	f = open(data_path + '/test_batch', 'rb')
	test_data_dict = pickle.load(f,encoding='bytes')
	f.close()
	XTest = np.array(test_data_dict[b"data"])
	yTest = np.array(test_data_dict[b"labels"])
	totaldata = np.concatenate((X,XTest),axis=0)
	totallabel =np.concatenate((y,yTest),axis=0)

	
	# print(X.shape,y.shape,XTest.shape,yTest.shape,type(X),type(y),type(XTest),type(yTest)) (50000, 3072) (50000,) (10000, 3072) (10000,)
	return totaldata,totallabel

def readMNIST(root):
	mnist=datasets.MNIST(root,download=True)
	# X,XTest,y,yTest=train_test_split(MNISTdataset.data,MNISTdataset.targets,test_size=0.2)
	mnist.data=mnist.data.numpy()
	mnist.targets=mnist.targets.numpy()
	return mnist.data, mnist.targets

def readtinyImageNet(root):
    tinyimagenet_tra,tinyimagenet_te=load_dataset('tinyImageNet')
    tinyimagenet_data=np.concatenate([tinyimagenet_tra.images,tinyimagenet_te.val_images],axis=0)
    tinyimagenet_targets=np.concatenate([tinyimagenet_tra.tra_labels,tinyimagenet_te.val_labels],axis=0)
    print("tinyimagenet:",tinyimagenet_data.shape,tinyimagenet_targets.shape)
    return tinyimagenet_data,tinyimagenet_targets 

def readGTSRB(root):
    gtsrbdataset=load_dataset('GTSRB')
    return gtsrbdataset.images,gtsrbdataset.targets

def shuffleAndSplitData(opt,dataX, dataY,cluster):
	num_shadow=opt.num_shadow
	# print(type(dataX))  nparray
	# print(type(dataY))
	c = zip(dataX, dataY)
	d=list(c)
	random.shuffle(d)
	dataX, dataY = zip(*d) #unpack
	# label=range(10)
	# selected_classes=np.random.choice(label,opt.classes,replace=False) 
	# selected_classes=label[:opt.classes]
	# print('***********************')
	# print(selected_classes)
	# print(type(dataX)) #tupe
	# print(type(dataY))
	dataX,dataY=np.array(dataX),np.array(dataY)
	# indexls = [i for i in range(dataY.shape[0]) if dataY[i] in selected_classes ]
	# index=np.array(indexls)
	# datax=dataX[index]
	# datay=dataY[index] 
	# print(datax.shape) #selected classes  classes=3: (18000,3072)
	# print(datay.shape)
	# print(np.array(dataX).shape) #total  (60000,3072)
	# dataX=datax
	# dataY=datay
	toTrainData  = np.array(dataX[:cluster]) 
	toTrainLabel = np.array(dataY[:cluster])

	toTestData  = np.array(dataX[cluster:cluster*2])
	toTestLabel = np.array(dataY[cluster:cluster*2])

	toTrainDataSave, toTestDataSave    = preprocessing(opt.dataset,toTrainData, toTestData)
	index=np.arange(len(dataY))

	shadowTrainindex=index[cluster*2:cluster*2+(len(dataY)-2*cluster)//2] #除去target train test之外的作为shadow models的train test来源
	shadowTestindex=index[cluster*2+(len(dataY)-2*cluster)//2:len(dataY)]

	totalshadowtrain=[]
	totalshadowtrainlabel=[]
	totalshadowtest=[]
	totalshadowtestlabel=[]
	for i in range(num_shadow):
		shadowDatatr_index=np.random.choice(shadowTrainindex,cluster,replace=False)
		shadowDatatr  = np.array(dataX)[shadowDatatr_index,:]
		shadowLabeltr = np.array(dataY)[shadowDatatr_index]
		
		shadowDatate_index=np.random.choice(shadowTestindex,cluster,replace=False)
		shadowTestData  = np.array(dataX)[shadowDatate_index,:]
		shadowTestLabel = np.array(dataY)[shadowDatate_index]
		shadowTrainDataSave, shadowTestDataSave = preprocessing(opt.dataset,shadowDatatr, shadowTestData)

		totalshadowtrain.append(shadowTrainDataSave)
		totalshadowtrainlabel.append(shadowLabeltr)
		totalshadowtest.append(shadowTestDataSave)
		totalshadowtestlabel.append(shadowTestLabel)
	totalshadowtrain=np.array(totalshadowtrain)
	totalshadowtrainlabel=np.array(totalshadowtrainlabel)
	totalshadowtest=np.array(totalshadowtest)
	totalshadowtestlabel=np.array(totalshadowtestlabel)
	print("totalshadowtraindata:",totalshadowtrain.shape)
	print("totalshadowtrainlabel:",totalshadowtrainlabel.shape)
	print("totalshadowtestdata:",totalshadowtest.shape)
	print("totalshadowtestlabel:",totalshadowtestlabel.shape)
	
	return toTrainDataSave, toTrainLabel,totalshadowtrain,totalshadowtrainlabel,toTestDataSave,toTestLabel,totalshadowtest,totalshadowtestlabel

def initializeData(args,root,num_shadow,dataFolderPath = './data/'):
    if(args.dataset == 'cifar10'):
        print("Loading data")
        dataX, dataY= readCIFAR10(root+'/cifar-10-batches-py')
        print("Preprocessing data")
        cluster = args.cluster  #2500 5000 10000 15000  #cluster : different training size
        dataPath = dataFolderPath+args.dataset+'/Preprocessed'
        toTrainData, toTrainLabel,totalshadowtrain,totalshadowtrainlabel,toTestData,toTestLabel,totalshadowtest,totalshadowtestlabel = shuffleAndSplitData(args,dataX, dataY,cluster)
        
    elif (args.dataset=='MNIST'):
        print("Loading MNIST data")
        dataX, dataY= readMNIST(root)
        print("Preprocessing data")
        cluster = args.cluster#12000 #total 60000
        dataPath = dataFolderPath+args.dataset+'/Preprocessed'
        toTrainData, toTrainLabel,totalshadowtrain,totalshadowtrainlabel,toTestData,toTestLabel,totalshadowtest,totalshadowtestlabel = shuffleAndSplitData(args,dataX, dataY,cluster)

    elif (args.dataset=='tinyImageNet'):
        print("Loading data")
        dataX, dataY= readtinyImageNet(root)
        print("Preprocessing data")
        cluster = args.cluster#20000  #total train: 10 0000
        dataPath = dataFolderPath+args.dataset+'/Preprocessed'
        toTrainData, toTrainLabel,totalshadowtrain,totalshadowtrainlabel,toTestData,toTestLabel,totalshadowtest,totalshadowtestlabel = shuffleAndSplitData(args,dataX, dataY,cluster)

    elif (args.dataset=='GTSRB'):
        print("Loading data")
        dataX, dataY= readGTSRB(root)
        print("Preprocessing data")
        cluster = args.cluster#8000  #total train: 39209
        dataPath = dataFolderPath+args.dataset+'/Preprocessed'
        toTrainData, toTrainLabel,totalshadowtrain,totalshadowtrainlabel,toTestData,toTestLabel,totalshadowtest,totalshadowtestlabel = shuffleAndSplitData(args,dataX, dataY,cluster)


    try:
        os.makedirs(dataPath+ '/'+str(args.cluster))

    except OSError:
        # assert(0)
        pass

    np.savez(dataPath + '/'+str(args.cluster)+'/targetTrain.npz', toTrainData, toTrainLabel)
    np.savez(dataPath + '/'+str(args.cluster)+'/targetTest.npz',  toTestData, toTestLabel)
    np.savez(dataPath + '/'+str(args.cluster)+'/shadowTrain.npz', totalshadowtrain, totalshadowtrainlabel)
    np.savez(dataPath + '/'+str(args.cluster)+'/shadowTest.npz',  totalshadowtest,totalshadowtestlabel)
    print("Preprocessing finished\n\n")
    return toTrainData, toTrainLabel, toTestData, toTestLabel, totalshadowtrain, totalshadowtrainlabel, totalshadowtest, totalshadowtestlabel


def preprocessing(dataset,toTrainData,toTestData):
	if dataset=='cifar10':
			def reshape_for_save(raw_data):
				raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
				raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3)).transpose(0,3,1,2)
				return raw_data.astype(np.float32)

			offset = np.mean(reshape_for_save(toTrainData), 0)
			scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

			def rescale(raw_data):
				return (reshape_for_save(raw_data) - offset) / scale

			return rescale(toTrainData), rescale(toTestData)

	elif dataset=='MNIST':
		toTrainData=np.expand_dims(toTrainData,axis=1)  #(12000,1,28,28)
		toTestData=np.expand_dims(toTestData,axis=1)
		toTestData=toTestData.astype(np.float32)
		toTrainData=toTrainData.astype(np.float32)
		offset=np.mean(toTrainData,0)
		scale=np.std(toTrainData,0)
		
		return (toTrainData-offset)/(scale+0.001),(toTestData-offset)/(scale+0.001)  

	elif dataset=='tinyImageNet':
		toTrainData=np.transpose(toTrainData,[0,3,1,2])
		toTestData=np.transpose(toTestData,[0,3,1,2])
		toTrainData=toTrainData.astype(np.float32)
		toTestData=toTestData.astype(np.float32)
		offset = np.mean(toTrainData, 0)
		scale  = np.std(toTrainData, 0).clip(min=1)
		return (toTrainData-offset)/(scale),(toTestData-offset)/(scale)
	
	elif dataset=='GTSRB':
		toTrainData=np.transpose(toTrainData,[0,3,1,2])
		toTestData=np.transpose(toTestData,[0,3,1,2])
		toTrainData=toTrainData.astype(np.float32)
		toTestData=toTestData.astype(np.float32)
		offset = np.mean(toTrainData, 0)
		scale  = np.std(toTrainData, 0).clip(min=1)
		return (toTrainData-offset)/(scale),(toTestData-offset)/(scale)   		
