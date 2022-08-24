import os
import  PIL
import numpy as np
import torch.utils.data as data
from torchvision import transforms

class ImageFolder(data.Dataset):
    def __init__(self, file_path,args):
        self.img_path = "../dataset/celeba_imgs"
        self.model_name = "target"
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = int(args.classnumber)
        print("Load " + str(self.num_img) + "target images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            img_name, iden = line.strip().split(' ')
            name_list.append(img_name)
            label_list.append(int(iden)-1)

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('RGB')
                img_list.append(img)
        return img_list
    
    def get_processor(self):
        if self.model_name == "FaceNet":
            re_size = 112
        else:
            re_size = 64
            
        crop_size = 108
        
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        label = self.label_list[index]
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1
        return img,  label

    def __len__(self):
        return self.num_img

class ImageFolder1(data.Dataset):
    def __init__(self, file_path,args):
        self.img_path = "../dataset/celeba_imgs"
        self.model_name = "target"
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path,args) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = int(args.classnumber)
        print("Load " + str(self.num_img) + " shadow images")


    def get_list(self, file_path,args):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            img_name, iden = line.strip().split(' ')
            name_list.append(img_name)
            label_list.append(int(iden)-int(args.classnumber)-1)
        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('RGB')
                img_list.append(img)
        return img_list
    
    def get_processor(self):
        if self.model_name == "FaceNet":
            re_size = 112
        else:
            re_size = 64
            
        crop_size = 108
        
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        label = self.label_list[index]

        return img,  label

    def __len__(self):
        return self.num_img




    

