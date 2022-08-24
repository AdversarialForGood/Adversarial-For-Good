from __future__ import print_function
import argparse 
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as Data 
import numpy as np
import classifier
from torchvision.models import AlexNet
from getdata import *
from MembershipInference.models import *
from utils import  init_dataloader, init_dataloader1
import warnings
warnings.filterwarnings("ignore")
def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def get_dataset(args):
    
    if (args.preprocess):
        root='../datasets'
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


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    description = "loss={:.4f} acc={:.2f}%"
    total = 0

    with tqdm(train_loader) as batch:
        
            for idx, (data, target) in enumerate(batch):
                optimizer.zero_grad()
                data, target = data.cuda(), target.cuda()
                output = model(data)
                target=target.to(torch.int64)
                loss = F.cross_entropy(output, target)
                total += data.shape[0]
                total_loss += loss.detach().item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                batch.set_description(description.format(total_loss / total, 100*correct / total))
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
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
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


def trainAttackModel(X_train, y_train, X_test, y_test):
    dataset = (X_train.astype(np.float32),
            y_train.astype(np.int32),
            X_test.astype(np.float32),
            y_test.astype(np.int32))

    output = classifier.train_model(dataset=dataset,
                                epochs=50,
                                batch_size=10,
                                learning_rate=0.01,
                                n_hidden=64,
                                l2_ratio = 1e-6,
                                model='softmax')

    return output


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,help='batch size for training target and shadow model')
    parser.add_argument('--epochs', type=int, default=20,help='epochs for training target and shadow model')
    parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--schedule',default=[25,40,50,75],help='when to change lr')
    parser.add_argument('--num_shadow',type=int,default=1,help='shadow model numbers')
    parser.add_argument('--cluster',type=int,default=2500,help='training and testing size of the target and shadow model, cifar:[2500,5000,7500,10000,12500,15000; celeba:[2,4,6,8,10]')
    parser.add_argument('--train', default=0, action='store_true',help='whether to train the target and shadow models')
    parser.add_argument('--weightdecay',type=float,default=5e-4,help='l2 regularization')
    parser.add_argument('--preprocess', action='store_true', help='preprocess the data, if false then load preprocessed data')
    parser.add_argument('--dataset',default='celeba',help='experiment dataset: cifar10 or celeba')
    parser.add_argument('--clip',type=int,default=3,help='clip digits of the prediction vector')
    parser.add_argument('--classes',type=int,default=10,help='number of classes')
    parser.add_argument('--structureT',default='vgg13',help='the target model structure')
    parser.add_argument('--structureS',default='vgg16',help='the shadow model structure')
    args = parser.parse_args()
    root='./result/model_based'
    datapath=root+'/data'
    modelpath=root+'/model'
    if args.dataset == 'celeba': #exp on celeba
        targettrain = './celeba_split/direct_split/'+args.classes+'_'+args.cluster+'/target_train.txt'
        targettest = './celeba_split/direct_split/'+args.classes+'_'+args.cluster+'/target_test.txt'
        shadowtrain = './celeba_split/direct_split/'+args.classes+'_'+args.cluster+'/shadow_train.txt'
        shadowtest = './celeba_split/direct_split/'+args.classes+'_'+args.cluster+'/shadow_test.txt'
        trainT = init_dataloader( targettrain, args)  
        testT = init_dataloader( targettest, args)
        trainS = init_dataloader1( shadowtrain, args)
        testS = init_dataloader1( shadowtest, args)
    
    else:  #exp on cifar10
        trainT, testT, trainS, testS=get_dataset(args)  


    thisdatapath=datapath+'/'+args.dataset+'/'+args.classnumber+'_'+args.cluster
    thismodelpath=modelpath+'/'+args.dataset+'/'+args.classnumber+'_'+args.cluster
    os.makedirs(thisdatapath,exist_ok=True)
    os.makedirs(thismodelpath,exist_ok=True)


    if args.train:
        # train target
        print('training target model...')
        if args.dataset=='cifar10':
            model = Net2(10).cuda()

        elif args.dataset == 'celeba':
            if args.structureT == 'vgg16':
                model = VGG16(int(args.classnumber)).cuda()
            elif args.structureT == 'vgg13':
                model = VGG13(int(args.classnumber)).cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        best_acc = -1

        train_loaderT = Data.DataLoader(trainT,
            batch_size=64, shuffle=True, num_workers=4)
        test_loaderT = Data.DataLoader(testT,
            batch_size=64, shuffle=True, num_workers=4)
        for epoch in tqdm(range(1, args.epochs + 1)):

            print('Epoch:',epoch)
            adjust_learning_rate(optimizer, epoch, args)
            train(model, train_loaderT, optimizer)
            acc, loss = test(model, test_loaderT)
            if acc > best_acc:
                torch.save(model.state_dict(), thismodelpath+'/targetmodel.pt')
            best_acc = max(best_acc, acc)

        # get attack model training data
        attack_x, attack_y = [], []
        
        if args.dataset=='cifar10':
            model = Net2(10).cuda()
        elif args.dataset == 'celeba':
            if args.structureT == 'vgg16':
                model = VGG16(int(args.classnumber)).cuda()
            elif args.structureT == 'vgg13':
                model = VGG13(int(args.classnumber)).cuda()
        model.load_state_dict(torch.load(thismodelpath+'/targetmodel.pt'))

        for data,target in train_loaderT:
            data,target=data.cuda(),target.cuda()
            attack_x.append(clipDataTopX((torch.softmax(model(data),dim=1)).cpu().detach().numpy(),args.clip))
            attack_y.append(np.ones(target.shape[0]))
            
        for data,target in test_loaderT:
            data,target=data.cuda(),target.cuda()
            attack_x.append(clipDataTopX((torch.softmax(model(data),dim=1)).cpu().detach().numpy(),args.clip))
            attack_y.append(np.zeros(target.shape[0]))

        attack_x = np.vstack(attack_x)
        attack_y = np.concatenate(attack_y)
        attack_x = attack_x.astype('float32')
        attack_y = attack_y.astype('int32') 
        np.savez(thisdatapath+'/targetmodeldata.npz',attack_x, attack_y)


        #train shadow
        print('training shadow model...')
        for i in range(args.num_shadow):

            train_loaderS = Data.DataLoader(trainS,
            batch_size=256, shuffle=True, num_workers=4)
            test_loaderS = torch.utils.data.DataLoader(testS,
            batch_size=256, shuffle=True, num_workers=4)
            
            if args.dataset=='cifar10':
                model= Net2(args.classes).cuda()
            elif args.dataset == 'celeba':
                if args.structureS == 'vgg16':
                    model = VGG16(int(args.classnumber)).cuda()
                elif args.structureS == 'vgg13':
                    model = VGG13(int(args.classnumber)).cuda()
                elif args.structureS == 'alexnet':
                    model = AlexNet(num_classes=1000).cuda()

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weightdecay)
            best_acc = -1

            for epoch in tqdm(range(1, args.epochs + 1)):

                print('Epoch:',epoch)
                adjust_learning_rate(optimizer, epoch, args)
                train(model, train_loaderS, optimizer)
                acc, loss = test(model, test_loaderS)
                if acc > best_acc:
                    torch.save(model.state_dict(), thismodelpath+'/number'+str(i)+'model.pt')
                best_acc = max(best_acc, acc)
            attack_x, attack_y = [], []
            if args.dataset=='cifar10':
                model = Net2(10).cuda()
            elif args.dataset == 'celeba':
                if args.structureS == 'vgg16':
                    model = VGG16(int(args.classnumber)).cuda()
                elif args.structureS == 'vgg13':
                    model = VGG13(int(args.classnumber)).cuda()
            model.load_state_dict(torch.load(thismodelpath+'/number'+str(i)+'model.pt'))

            for data,target in train_loaderS:
                data,target=data.cuda(),target.cuda()
                attack_x.append(clipDataTopX((torch.softmax(model(data),dim=1)).cpu().detach().numpy(),args.clip))
                attack_y.append(np.ones(target.shape[0]))
                
            for data,target in test_loaderS:
                data,target=data.cuda(),target.cuda()
                attack_x.append(clipDataTopX((torch.softmax(model(data),dim=1)).cpu().detach().numpy(),args.clip))
                attack_y.append(np.zeros(target.shape[0]))

            attack_x = np.vstack(attack_x)
            attack_y = np.concatenate(attack_y)
            attack_x = attack_x.astype('float32')
            attack_y = attack_y.astype('int32')
            np.savez(thisdatapath+'/number'+str(i)+'shadowdata.npz',attack_x, attack_y)


    else: pass #load attack data directly

    # train attack model
    targetX, targetY  = load_data( thisdatapath+'/targetmodeldata.npz')
    totalshadowX=[]
    totalshadowY=[]

    for i in range(args.num_shadow):

        shadowX, shadowY = load_data(thisdatapath+'/number'+str(i)+'shadowdata.npz')
        totalshadowX.append(shadowX)
        totalshadowY.append(shadowY)
        
    totalshadowX=np.stack(totalshadowX,0)
    totalshadowY=np.stack(totalshadowY,0)
    totalshadowX=totalshadowX.reshape(totalshadowX.shape[0]*totalshadowX.shape[1],totalshadowX.shape[2])
    totalshadowY=totalshadowY.reshape(totalshadowY.shape[0]*totalshadowY.shape[1],)	
    print("attack model training size:",totalshadowX.shape)


    
    #celeba attack model
    if args.dataset == 'celeba':
        model = attack_celeba(args.clip).cuda()
    #cifar attack model
    else: model =attack_cifar10(args.clip).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
    best_acc = -1
    trainset=Data.TensorDataset(torch.from_numpy(totalshadowX),torch.from_numpy(totalshadowY))
    testset =Data.TensorDataset(torch.from_numpy(targetX),torch.from_numpy(targetY))
    train_loader = Data.DataLoader(trainset,
    batch_size=64, shuffle=True, num_workers=4)
    test_loader = Data.DataLoader(testset,
    batch_size=64, shuffle=True, num_workers=4)
    max_acc=-1
    for epoch in tqdm(range(1, 50)):
        print('epoch:',epoch)
        adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer) 
        acc, loss = test(model, test_loader)
        if acc>max_acc:
            max_acc=acc
    print("Best attack accuracy:",max_acc)
  



if __name__ == '__main__':

            main()