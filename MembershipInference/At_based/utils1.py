import torch.nn.init as init
import json, time, random, torch, math
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
import dataloader 
from scipy.signal import convolve2d

device = "cuda"



def init_dataloader( file_path, args):
    data_set = dataloader.ImageFolder( file_path,args)
    return data_set

def init_dataloader1( file_path, args):
    data_set = dataloader.ImageFolder1( file_path,args)
    return data_set

def init_dataloader1_alex( file_path, args):
    data_set = dataloader.ImageFolder1alex( file_path,args)
    return data_set

def init_dataloader_alex( file_path, args):
    data_set = dataloader.ImageFolderalex( file_path,args)
    return data_set