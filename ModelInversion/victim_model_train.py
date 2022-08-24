import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import os
import torch.nn.functional as F




def adjust_learning_rate(lr, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    for milestone in [10,15]:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Victim_T_training(opt,T, epochs, dataloader_celeba, testloader, save_model_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        T.parameters(),
    lr=0.001, weight_decay= 5e-5
    )	
    bestacc = -1
    name=str(opt.epoch)+'epoch_'+'t'+opt.clusterT +"celeba_T.tar"

    print('begin training t')
    os.makedirs(save_model_dir, exist_ok=True)
    for epoch in range(epochs):
        # print(epoch)
        sum_loss = 0.0
        T.train()
        for i, data in enumerate(dataloader_celeba):
            inputs, onehot, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad() 
            outputs = T(inputs) 
            loss = criterion(outputs[1], labels)
            loss.backward() 
            optimizer.step() 
            # print('i',i)
            sum_loss += loss.item()
            if i % 200 == 0:
                print('Target Training Epoch [%d] loss:%.03f' %
                (epoch + 1, sum_loss / 100))
                sum_loss = 0.0
        T.eval()
        len_target_dataset = 0
        correct = 0
        for i, data in enumerate(dataloader_celeba):
            inputs, onehot, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = T(inputs)[1]
            eval_iden = torch.argmax(outputs, 1).view(-1)
            
            for i in range(len(labels)):
                gt = labels[i].item()
                if eval_iden[i].item() == gt:
                    correct += 1
            len_target_dataset += len(labels)
        acc = correct * 1.0 / len_target_dataset
        print('Epoch {}, training acc:{}'.format(epoch+1,acc))
        testacc, loss = test(T, testloader)

        if (epoch % 5 ==0):
            print("Test acc:{}".format(testacc))
        if testacc > bestacc:
             bestacc = testacc
             torch.save({'state_dict':T.state_dict()}, os.path.join(save_model_dir, name))
    print('Best test acc:{}'.format(bestacc))
    return T

def Victim_E_training(opt,E, epochs, dataloader_celeba, testloader, save_model_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    E.parameters(),
    lr=0.001,  weight_decay= 5e-5
    )
    bestacc = -1
    for epoch in range(epochs):
        
        sum_loss = 0.0
        E.train()
        for i, data in enumerate(dataloader_celeba):
            inputs, onehot, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad() 
            outputs = E(inputs) 
            loss = criterion(outputs[1], labels)
            loss.backward() 
            optimizer.step() 
            sum_loss += loss.item()
            if i % 200 == 0:
                print('Evaluation Training Epoch [%d] loss:%.03f' %
                (epoch + 1,  sum_loss / 100))
                sum_loss = 0.0

        E.eval()
        len_target_dataset = 0
        correct = 0
        for i, data in enumerate(dataloader_celeba):
            inputs, onehot, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = E(inputs)[1]
            eval_iden = torch.argmax(outputs, 1).view(-1)
            
            for i in range(len(labels)):
                gt = labels[i].item()
                if eval_iden[i].item() == gt:
                    correct += 1
            len_target_dataset += len(labels)
        acc = correct * 1.0 / len_target_dataset
        print('Epoch {}, training acc:{}'.format(epoch+1,acc))

        name=str(opt.epoch)+'epoch_'+'t'+opt.clusterT+"celeba_E.tar"
        testacc, loss = test(E, testloader)

        if (epoch % 2 ==0):
            print("Test acc:{}".format(testacc))
        if testacc > bestacc :
            bestacc = testacc
            torch.save({'state_dict':E.state_dict()}, os.path.join(save_model_dir,name))
    print('Best test acc:{}'.format(bestacc))

    return E


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data,_, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            target=target.to(torch.int64)
            test_loss += F.cross_entropy(output[1], target, size_average=False).item()  # sum up batch loss
            # test_loss +=  pt_categorical_crossentropy(output,target)
            total += data.shape[0]
            pred = torch.max(output[1], 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss