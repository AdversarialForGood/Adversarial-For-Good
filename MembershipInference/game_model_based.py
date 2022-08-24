
from __future__ import print_function
import argparse  
import torch
import torch.nn.functional as F
from torchvision import  transforms
import torchvision
import torch.utils.data as Data 
import numpy as np
from getdata import *
from MembershipInference.models import *
from utils import init_dataloader,init_dataloader1

def clipDataTopX(dataToClip, top=10):
    res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
    return np.array(res)

def get_dataset(args):
    cluster=args.cluster
    if args.split_type=='torch':
        trainset=torchvision.datasets.CIFAR10('../datasets', train=True, download=True,
                            transform=transforms.Compose(
                                [

                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))

        testset=torchvision.datasets.CIFAR10('../datasets', train=False, download=True, 
                            transform=transforms.Compose(
                                [
    
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))

        totalset=Data.ConcatDataset([trainset,testset])
        trainT,testT,trainS,testS=Data.random_split(dataset=totalset,lengths=[cluster,cluster,int((60000-2*cluster)/2),int((60000-2*cluster)/2)])

    elif args.split_type=='lasagne':
        if (args.preprocess):
            root='/home/baiyj/sok/datasets'
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
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss


def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--schedule',default=[25,40,50,75],help='when to change lr')
    parser.add_argument('--num_shadow',type=int,default=1)
    parser.add_argument('--cluster',type=int,default=2500)
    parser.add_argument('--train', default=0, action='store_true',help='whether to train the target and shadow models')
    parser.add_argument('--weightdecay',type=float,default=5e-4,help='l2 regularization')
    parser.add_argument('--split_type',type=str,default='lasagne',help="how to prepare the target and shadow training data")
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data, if false then load preprocessed data')
    parser.add_argument('--dataset',default='celeba')
    parser.add_argument('--clip',type=int,default=1000)
    parser.add_argument('--classes',type=int,default=1000)
    parser.add_argument('--defend',type=int,default=1)
    args = parser.parse_args()

   
    if args.dataset == 'celeba':
        targetmodel = VGG16(1000).cuda()
        attackmodel = attack_celeba(args.clip).cuda()

    else:
        targetmodel=Net1(10).cuda()
        attackmodel=attack_cifar10(10).cuda()

    #load a trained attack model
    print('load attack model..')
    attackmodel=torch.load('./result/model_based/model/celeba/attackmodel.pt')

    #load target model
    if args.defend ==1:
        print('load defended model..')
        checkpoint=torch.load("./AdversarialTraining/results/checkpoints_celeba/epoch49alpha5.0attackoriattack")
        targetmodel.load_state_dict(checkpoint['state_dict'])
    else:
        print('load normal target model..')        
        targetmodel.load_state_dict(torch.load('./result/model_based/model/2500/targetmodel.pt'))
    if (args.dataset == 'celeba'):
        targettrain = './celeba_split/direct_split/1000_10/target_train.txt'
        targettest = './celeba_split/direct_split/1000_10/target_test.txt'
        shadowtrain = './celeba_split/direct_split/1000_10/shadow_train.txt'
        shadowtest = './celeba_split/direct_split/1000_10/shadow_test.txt'
        targetT, _ = init_dataloader( targettrain, 16)   
        testT, _ = init_dataloader( targettest, 16)
        trainS, _ = init_dataloader1( shadowtrain, 16)
        testS, _ = init_dataloader1( shadowtest, 16)
    else:
        targetT, testT, _, _=   get_dataset(args)
    train_loaderT = Data.DataLoader(targetT,
    batch_size=256, shuffle=True, num_workers=4)
    test_loaderT = Data.DataLoader(testT,
    batch_size=256, shuffle=True, num_workers=4)

    attack_x=[]
    attack_y=[]
    for data,target in train_loaderT:
        data,target=data.cuda(),target.cuda()
        attack_x.append(clipDataTopX((torch.softmax(targetmodel(data),dim=1)).cpu().detach().numpy(),args.clip))
        attack_y.append(np.ones(target.shape[0]))
        
    for data,target in test_loaderT:
        data,target=data.cuda(),target.cuda()
        attack_x.append(clipDataTopX((torch.softmax(targetmodel(data),dim=1)).cpu().detach().numpy(),args.clip))
        attack_y.append(np.zeros(target.shape[0]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32') 
    testset =Data.TensorDataset(torch.from_numpy(attack_x),torch.from_numpy(attack_y))
    test_loader = Data.DataLoader(testset,batch_size=64, shuffle=True, num_workers=4)
    test(attackmodel,test_loader)

main()