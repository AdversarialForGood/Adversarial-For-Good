import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import os
import torch.nn.functional as F

def Victim_T_training_at(opt,T, G, epochs, dataloader_celeba, save_model_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        T.parameters(),
    lr=0.001,
    )	
    os.makedirs(save_model_dir, exist_ok=True)
    for epoch in range(epochs):
        sum_loss = 0.0
        T.train()
        for i, data in enumerate(dataloader_celeba):
            inputs, onehot, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad() 
            outputs = T(inputs)

            #G loss
            G.eval()
            rec = G(F.softmax(outputs[1],dim=1))
            loss_G = F.mse_loss(rec, inputs)
            
            
            #T loss
            loss = criterion(outputs[1], labels)

            total_loss = loss - opt.lamda*loss_G
            total_loss.backward() 
            optimizer.step() 

            sum_loss += loss.item()
            # if i % 100 == 99:
            #     # print('[%d,%d] loss:%.03f' %
            #     (epoch + 1, i + 1, sum_loss / 100))
            #     sum_loss = 0.0
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
        # print('training acc:',acc)
        name=str(opt.epoch)+'epoch_'+'t'+opt.clusterT +"celeba_T.tar"
    torch.save({'state_dict':T.state_dict()}, os.path.join(save_model_dir, name))
    return T



def Victim_T_training(opt, T,  epochs, dataloader_celeba, save_model_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        T.parameters(),
    lr=0.001,
    )	
    os.makedirs(save_model_dir, exist_ok=True)
    for epoch in range(epochs):
        sum_loss = 0.0
        T.train()
        for i, data in enumerate(dataloader_celeba):
            inputs, onehot, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad() 
            outputs = T(inputs)
            
            #T loss
            loss = criterion(outputs[1], labels)

            total_loss = loss
            total_loss.backward() 
            optimizer.step() 

            sum_loss += loss.item()
            # if i % 100 == 99:
            #     # print('[%d,%d] loss:%.03f' %
            #     (epoch + 1, i + 1, sum_loss / 100))
            #     sum_loss = 0.0
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
        # print('training acc:',acc)
        name=str(opt.epoch)+'epoch_'+'t'+opt.clusterT +"celeba_T.tar"
    torch.save({'state_dict':T.state_dict()}, os.path.join(save_model_dir, name))
    print("Finish training the target model")
    return T


def Victim_E_training(opt,E, epochs, dataloader_celeba, save_model_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    E.parameters(),
    lr=0.001,
    )
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
            # if i % 100 == 99:
            #     print('[%d,%d] loss:%.03f' %
            #     (epoch + 1, i + 1, sum_loss / 100))
            #     sum_loss = 0.0

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
        # print('training E acc:',acc)
        name=str(opt.epoch)+'epoch_'+'t'+opt.clusterT+"celeba_E.tar"
    torch.save({'state_dict':E.state_dict()}, os.path.join(save_model_dir,name))
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