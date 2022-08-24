from sklearn.utils import shuffle
import torch
import torch.utils.data as Data
from torchvision import datasets,transforms
import ast
import configparser
import numpy as np
class InputData():
    def __init__(self,dataset):
        self.dataset=dataset
        self.data_filepath='../../datasets'
        def __init__(self, dataset):
            self.dataset=dataset

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.user_training_number=2500
        self.user_test_number=2500
        self.defender_member_number=2500
        self.defender_nonmember_number=2500
        self.attacker_evaluate_member_number=2500
        self.attacker_evaluate_nonmember_number=2500
        self.attacker_train_member_number=2500
        self.attacker_train_nonmember_number=2500
        if self.dataset=='cifar10':
            trainset=datasets.CIFAR10(root=self.data_filepath,train=True,transform=transforms.Compose(
                                [
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))
        
            testset=datasets.CIFAR10(root=self.data_filepath,train=False,transform=transforms.Compose(
                                [
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))
        
        else:
            pass
        self.trainset=trainset
        self.testset=testset
        self.res=len(self.trainset.targets)-self.user_training_number- self.user_test_number-self.attacker_evaluate_nonmember_number-self.attacker_train_member_number-self.attacker_train_nonmember_number


    def split(self):
        train_user,test_user,attacker_evaluate_nonmember,trainmem_attacker,trainnon_attacker,_ = Data.random_split(
                dataset=self.trainset,
                lengths=[self.user_training_number, self.user_test_number, self.attacker_evaluate_nonmember_number, self.attacker_train_member_number,self.attacker_train_nonmember_number,self.res],
                generator=torch.Generator().manual_seed(0)
            )
        return train_user,test_user,attacker_evaluate_nonmember,trainmem_attacker,trainnon_attacker
      
    def input_data_user(self):
        train_user,test_user,_,_,_=self.split(self)
        return train_user,test_user
    
    def input_data_defender(self):
        train_user,test_user,_,_,_=self.split(self)
        x_train_defender=np.concatenate((train_user.data,test_user.data))
        y_train_defender=np.concatenate((train_user.targets),test_user.targets)
        train_defender=Data.TensorDataset(x_train_defender,y_train_defender)
        label_train_defender=np.zeros(x_train_defender.shape[0])
        label_train_defender[0:train_user.data.shape[0]]=1
        
        return train_defender,label_train_defender

    def input_data_attacker_adv1(self):
        _,_,_,train_member_attacker,train_nonmember_attacker,_=self.split(self)
        x_train_attacker=np.concatenate((train_member_attacker.data),(train_nonmember_attacker.data))
        y_train_attacker=np.concatenate((train_member_attacker.targets),(train_nonmember_attacker.targets))
        train_attacker=Data.TensorDataset(x_train_attacker,y_train_attacker)
        label_train_attacker=np.zeros(x_train_attacker.shape[0])
        label_train_attacker[0:train_member_attacker.data.shape[0]]=1

        return train_attacker,label_train_attacker

    def input_data_attacker_evaluate(self):
        attacker_evaluate_member,_,attack_evaluate_non,_,_=self.split(self)
        x_evaluate_attacker=np.concatenate(attacker_evaluate_member.data,attack_evaluate_non.data)
        y_evaluate_attacker=np.concatenate(attacker_evaluate_member.targets,attack_evaluate_non.targets)
        evaluate_attacker=Data.TensorDataset(x_evaluate_attacker,y_evaluate_attacker)
        label_evaluate_attacker=np.zeros(x_evaluate_attacker.shape[0])
        label_evaluate_attacker[0:attacker_evaluate_member.data.shape[0]]=1

        return evaluate_attacker,label_evaluate_attacker

    def input_data_attacker_shallow_model_adv1(self):
         _,_,_,train_member_attacker,train_nonmember_attacker,_=self.split(self)
         return train_member_attacker,train_nonmember_attacker
        
        
