import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn





#Cifar10 models
class Net1(nn.Module):  
    def __init__(self,classes):
        super(Net1,self).__init__()
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
        x = self.fc2(x)
        return x

class Net2(nn.Module): 
    def __init__(self,classes):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  



#Inference attack models
class attack_cifar10(nn.Module):  
    def __init__(self,clip=3):
        super(attack_cifar10, self).__init__()

        self.fc1 = nn.Linear(clip , 64)
        self.fc2 = nn.Linear(64, 100)
        self.fc3 = nn.Linear(64,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  

class attack_celeba(nn.Module):  
    def __init__(self,clip=3):
        super(attack_celeba, self).__init__()

        self.fc1 = nn.Linear(clip , 2000)
        self.fc2 = nn.Linear(2000, 1500)
        self.fc3 = nn.Linear(1500,500)
        self.fc4 = nn.Linear(500,64)
        # self.drop = nn.Dropout(0.2)
        self.fc5 = nn.Linear(64,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = self.drop(x)
        x = self.fc5(x)
        return x  



#Celeba models
class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=False)
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

class VGG13(nn.Module):
    def __init__(self, n_classes):
        super(VGG13, self).__init__()
        model = torchvision.models.vgg13_bn(pretrained=False)
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

class VGG19(nn.Module):
    def __init__(self, n_classes):
        super(VGG19, self).__init__()
        model = torchvision.models.vgg19_bn(pretrained=False)
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