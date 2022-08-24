import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import  roc_curve, auc
from per import *
import argparse  
import os
from sklearn import cluster
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision
from tqdm import tqdm
import torch.utils.data as Data 
import numpy as np
from getdata import *
from MembershipInference.models import *
from utils import  init_dataloader

def attackbyorder(values,groundtruth):
    
    index=values.argsort()
    groundtruth=groundtruth[index]
    pred=np.zeros(groundtruth.shape[0])
    pred[:int((groundtruth.shape[0])/2)]=1
    correct=np.sum(pred==groundtruth)
    acc=correct/(groundtruth.shape[0]*1.0)
    return acc
	

def clipDataTopX(dataToClip, top=3):
    res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
    return np.array(res)

def getoutput(targetmodel,data_loader) :
    Toutput=[]
    targets=[]
    targetmodel.eval()


    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            Toutput.append(np.array(torch.softmax(targetmodel(data),dim=1).cpu()))
            targets.append(int(target.cpu()))
            
    Toutput=np.array(Toutput)
    print(Toutput.shape)
    Toutput=np.squeeze(Toutput)
    print(Toutput.shape)

    t_tr_outputs=Toutput[0:int(len(data_loader.dataset)/2),:]
    t_te_outputs=Toutput[int(len(data_loader.dataset)/2):len(data_loader.dataset),:]
   
    t_tr_labels=targets[0:int(len(data_loader.dataset)/2)]
    t_te_labels=targets[int(len(data_loader.dataset)/2):len(data_loader.dataset)]
   
    return  t_tr_outputs,t_te_outputs,t_tr_labels,t_te_labels


def Metric_based_acc(args,metric,targetmodel, data_loader):
    if metric=='perturbation':
        t_tr_dis,t_te_dis=AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster,  maxitr=50, max_eval=10000)
        np.savez(args.root+'/perturbation'+'/'+str(args.cluster)+'/perturbation.npz',t_tr_dis,t_te_dis)
        targets=[]
        with torch.no_grad():
            for data, target in data_loader:
                targets.append(int(target.cpu()))
        t_tr_labels=targets[0:int(len(data_loader.dataset)/2)]
        t_te_labels=targets[int(len(data_loader.dataset)/2):len(data_loader.dataset)]
        metric_attack(args,metric,10,None,None,t_tr_labels,t_te_labels,t_tr_dis,t_te_dis)
    
    else:
        t_tr_outputs,t_te_outputs,t_tr_labels,t_te_labels=getoutput(targetmodel,data_loader)
        metric_attack(args,metric,10,t_tr_outputs,t_te_outputs,t_tr_labels,t_te_labels,None,None)

def metric_attack(args,metric,num_classes,t_tr_outputs,t_te_outputs,t_tr_labels,t_te_labels,t_tr_dis,t_te_dis):
    def log_value( probs, small_value=1e-1):
        return -np.log(np.maximum(probs, small_value))
    
    def entr_comp( probs):
        entropy=np.sum(np.multiply(probs, log_value(probs)),axis=1)
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
        couple=zip(probs,true_labels)
        loss=[F.cross_entropy(torch.from_numpy(prob), torch.tensor(target)) for (prob,target) in couple]
        return np.array(loss)

    def correctness_based():
        t_tr_corr = (np.argmax(t_tr_outputs, axis=1)==t_tr_labels).astype(int)
        t_te_corr = (np.argmax(t_te_outputs, axis=1)==t_te_labels).astype(int)
        t_tr_acc = np.sum(t_tr_corr)/(len(t_tr_corr)+0.0)
        t_te_acc = np.sum(t_te_corr)/(len(t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('correctness-based MIA , the attack acc is {acc:.3f} with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return

    def confidence_based():
        t_tr_conf = np.array([t_tr_outputs[i, t_tr_labels[i]] for i in range(len(t_tr_labels))])
        t_te_conf = np.array([t_te_outputs[i, t_te_labels[i]] for i in range(len(t_te_labels))])
        truth1=np.ones(t_tr_conf.shape[0])
        truth2=np.zeros(t_te_conf.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        targetvalues=np.concatenate((t_tr_conf,t_te_conf))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=1, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)
        acc=attackbyorder(targetvalues,groundtruth[::-1])  
        print('confidence-based MIA, the attack acc is {acc:.3f}'.format(acc=acc))
        return
      
    def entropy_based():
        t_tr_entr = entr_comp(clipDataTopX(t_tr_outputs,args.clip))
        t_te_entr = entr_comp(clipDataTopX(t_te_outputs,args.clip))
        truth1=np.ones(t_tr_entr.shape[0])
        truth2=np.zeros(t_te_entr.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        targetvalues=np.concatenate((t_tr_entr,t_te_entr))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=0, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)
        acc=attackbyorder(targetvalues,groundtruth)
        print('entropy-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return    

    def modientropy_based():
        t_tr_entr = m_entr_comp(t_tr_outputs,t_tr_labels)
        t_te_entr = m_entr_comp(t_te_outputs,t_te_labels)
        truth1=np.ones(t_tr_entr.shape[0])
        truth2=np.zeros(t_te_entr.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        targetvalues=np.concatenate((t_tr_entr,t_te_entr))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=0, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)
        acc=attackbyorder(targetvalues,groundtruth)
        print('modified entropy-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return 

    def loss_based():
        t_tr_loss = loss(t_tr_outputs,t_tr_labels)
        t_te_loss = loss(t_te_outputs,t_te_labels)
        t_tr_mem, t_te_non_mem = 0, 0
        truth1=np.ones(t_tr_loss.shape[0])
        truth2=np.zeros(t_te_loss.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        print(groundtruth.shape)
        targetvalues=np.concatenate((t_tr_loss,t_te_loss))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=0, drop_intermediate=False)
        auc_loss = round(auc(fpr, tpr), 4)
        print('auc is:',auc_loss)
        acc=attackbyorder(targetvalues,groundtruth)
        print('loss-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return 

    def per_based():
        t_tr_per=t_tr_dis
        t_te_per=t_te_dis
        truth1=np.ones(t_tr_per.shape[0])
        truth2=np.zeros(t_te_per.shape[0])
        groundtruth=np.concatenate((truth1,truth2))
        targetvalues=np.concatenate((t_tr_per,t_te_per))
        fpr, tpr, _ = roc_curve(groundtruth, targetvalues, pos_label=1, drop_intermediate=False)
        auc_entro = round(auc(fpr, tpr), 4)
        print('auc is:',auc_entro)
        acc=attackbyorder(np.squeeze(targetvalues),groundtruth[::-1])
        print('perturbation-based MIA , the attack acc is {acc:.3f}'.format(acc=acc))
        return 


    if metric=='correctness': correctness_based()
    elif metric=='confidence': confidence_based()
    elif metric=='entropy':    entropy_based()
    elif metric=='modifiedentropy': modientropy_based()
    elif metric=='loss': loss_based()
    elif metric=='perturbation':per_based()
    else: assert(0)
    
def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def get_dataset(args):
   
    if (args.preprocess):
        root='/home/baiyj/sok/datasets'
        targetTrain, targetTrainLabel, targetTest, targetTestLabel, shadowTrain, shadowTrainLabel, shadowTest, shadowTestLabel=initializeData(args,root,args.num_shadow,dataFolderPath = './data/')            
        shadowTrain, shadowTrainLabel=np.squeeze(shadowTrain),np.squeeze(shadowTrainLabel)
        shadowTest, shadowTestLabel=np.squeeze(shadowTest),np.squeeze(shadowTestLabel)

    else: #load the preprocessed data
        targetTrain, targetTrainLabel  = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+ str(args.classes)+'/'+str(args.cluster) +'/targetTrain.npz')
        targetTest,  targetTestLabel   = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+ str(args.classes)+'/'+str(args.cluster) + '/targetTest.npz')
        shadowTrain, shadowTrainLabel  = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+ str(args.classes)+'/'+str(args.cluster) + '/shadowTrain.npz')
        shadowTest,  shadowTestLabel   = load_data('./data' +'/'+args.dataset+'/Preprocessed'+'/'+ str(args.classes)+'/'+str(args.cluster) + '/shadowTest.npz')
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--schedule',default=[25,40,50,75],help='when to change lr')
    parser.add_argument('--cluster',type=int,default=2500)
    parser.add_argument('--train', default=0, action='store_true',help='whether to train the target and shadow models')
    parser.add_argument('--weightdecay',type=float,default=5e-4,help='l2 regularization')
    parser.add_argument('--split_type',type=str,default='lasagne',help="how to prepare the target and shadow training data")
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data, if false then load preprocessed data')
    parser.add_argument('--dataset',default='cifar10')
    parser.add_argument('--metric',default='confidence',help='entropy,correctness,confidence,modifiedentropy,loss')
    parser.add_argument('--dis',default='l0',help="perturbation type")
    parser.add_argument('--clip',type=int,default=1000)
    parser.add_argument('--classes',type=int,default=10)
    parser.add_argument('--per', type=int, default=0)

    args = parser.parse_args()
    args.classnumber = args.classes
    root='./result/metric_based'
    args.root=root
    datapath=root+'/data'
    modelpath=root+'/model'
    if args.dataset == 'celeba': #exp on celeba
        print(type(args.classnumber))
        print(type(args.trainingsize))
        targettrain = './celeba_split/direct_split/'+str(args.classnumber)+'_'+str(args.trainingsize)+'/target_train.txt'
        targettest = './celeba_split/direct_split/'+str(args.classnumber)+'_'+str(args.trainingsize)+'/target_test.txt'
        trainT = init_dataloader( targettrain, args)  
        testT = init_dataloader( targettest, args)
    else:  #exp on cifar10
        trainT, testT, trainS, testS=get_dataset(args)
    attackdataset=Data.ConcatDataset([trainT,testT])  
    thismodelpath=modelpath+'/Epoch'+str(args.epochs) +'/Cluster'+str(args.cluster)+'/'+ str(args.classes)#+str(cluster)
    os.makedirs(thismodelpath,exist_ok=True)


    if args.train:
        #train target
        if args.dataset=='celeba':
            model=VGG16(int(args.classnumber)).cuda()
        elif args.dataset=='cifar10':
            model=Net1(int(args.classnumber)).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)#,weight_decay=args.weightdecay)
        best_acc = -1
        train_loaderT = Data.DataLoader(trainT,
            batch_size=256, shuffle=True, num_workers=4)
        test_loaderT = Data.DataLoader(testT,
            batch_size=256, shuffle=True, num_workers=4)
        for epoch in tqdm(range(1, args.epochs + 1)):
            adjust_learning_rate(optimizer, epoch, args)
            train(model, train_loaderT, optimizer)
            acc, loss = test(model, test_loaderT)
            if acc > best_acc:
                torch.save(model.state_dict(), thismodelpath+'/targetmodel.pt')
            best_acc = max(best_acc, acc)        

    else: pass #load attack data directly
    
    # load target 
    if args.dataset=='celeba':
        targetmodel=VGG16(int(args.classnumber)).cuda()
    elif args.dataset=='cifar10':
        targetmodel= Net1(int(args.classnumber)).cuda()

    targetmodel.load_state_dict(torch.load(thismodelpath+'/targetmodel.pt'))
    attackdataloader=Data.DataLoader(attackdataset,batch_size=1, shuffle=False, num_workers=4)
    Metric_based_acc(args,args.metric,targetmodel, attackdataloader)
  


if __name__ == '__main__':

            main()