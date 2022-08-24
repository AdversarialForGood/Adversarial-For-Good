import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
from torch.autograd import Variable
import time
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math
import json
import torch.optim as optim
import dataloader
import utils
from utils import *
import sys
from evaluation import feature_distance, knn_distance, target_model, evaluate
from inversion_method_at import adversarial_attack_at, gmi_attack, adversarial_attack
from victim_model_train_at import Victim_T_training, Victim_T_training_at, Victim_E_training, test
from torchvision import utils as vutils
import argparse  


device = "cuda"
num_classes = 1000
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', type=int, default=50)
	parser.add_argument('--Gepoch', type=str, default='100')

	parser.add_argument('--clusterG',type=str,default='10')
	parser.add_argument('--clusterT',type=str,default='10')
	parser.add_argument('--trainT',type=int, default=0)
	parser.add_argument('--trainE',type=int, default=0)
	parser.add_argument('--trainG',type=int, default=0)
	parser.add_argument('--attack',type=str,default='ad',help='ad or gmi')
	parser.add_argument('--mode',type=str,default='similar',help='same or similar or diff')
	parser.add_argument('--classnumberT',type=str, default= '1000')
	parser.add_argument('--classnumberG',type=str, default='1000')
	parser.add_argument('--pre_ad', type=int, default=0)
	parser.add_argument('--pre_gmi', type=int, default=0)
	parser.add_argument('--defend', type=int, default=0)
	parser.add_argument('--epsilon',type=float,default=0.3)
	parser.add_argument('--atepoch',type=int,default=10)
	parser.add_argument('--lamda',type=float,default=0.5)
	


	

	opt = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	file = "./celeba1.json"
	args = load_json(json_file=file)
	print(args)
	data_split_path = '../dataset/celeba_split/direct_split/'
	file_nameT = 'celeba_target_list_' + opt.classnumberT + '_t' +opt.clusterT +'.txt'
	if opt.mode == 'diff':
		file_nameG = 'celeba_generation_list_diff' + opt.classnumberG +'_g'+opt.clusterG + '.txt'
	elif opt.mode == 'similar':
		file_nameG = 'celeba_generation_list_' + opt.classnumberT +'_g'+opt.clusterG + '.txt' 
	elif opt.mode == 'same':
		file_nameG = file_nameT
	else: assert(0)
	file_pathT = data_split_path + file_nameT
	file_pathG = data_split_path + file_nameG


    		
	model_name = 'TARGET'
	lr = 0.001
	batch_size = 64
	epochs = opt.epoch

	#T , E  dataset  and  G dataset (for test T E)
	dataset_celeba, dataloader_celeba = init_dataloader(args, int(opt.classnumberT), file_pathT, batch_size, mode="target")
	dataset_G, dataloader_G = init_dataloader(args , int(opt.classnumberT) , file_pathG ,batch_size, mode='target')

	
	T = classify.VGG16(int(opt.classnumberT))
	T = T.cuda()
	# training the target model and the evaluation model
	save_model_dir = "./victim_model_at/base_epoch_100/" + 'Epoch' +str(opt.epoch) + 'Cluster' + opt.clusterT + 'Class' + opt.classnumberT + 'lamda' + str(opt.lamda) +  '/'
	os.makedirs(save_model_dir, exist_ok= True)

	print("------------Loading Evaluation----------------")
	E = classify.VGG16(int(opt.classnumberT))
	E = E.cuda()
	E.load_state_dict(torch.load('/victim_model/base_epoch_100/140epoch_t10celeba_E.tar') ['state_dict'] )


	T_model = target_model(T)
	iden = torch.zeros(30)
	for i in range(30):
		iden[i] = i
	
	print("-----------Begin adversarial training------------")

	if opt.attack == 'ad':
		print('=========================number 0 epoch adversarial training...=========================')
		print('---------------------Training target model-----------------------------')
		T = Victim_T_training(opt, T , int(epochs/opt.atepoch), dataloader_celeba, save_model_dir)
		print("---------------------Training the generation model---------------------")
		_, G = adversarial_attack(opt, int(int(opt.Gepoch)/opt.atepoch),  file_pathT, file_pathG, data_split_path=data_split_path, target_dataset='celeba', T=T_model, E=E, iden=iden, pretrained=0 , test =0)
    	
		for i in range(opt.atepoch-1) :  
			print('number{} epoch adversarial training...'.format(i+1))
			print("---------------------Training the generation model---------------------")
			G = adversarial_attack_at(opt, int(int(opt.Gepoch)/opt.atepoch),file_pathT, file_pathG, data_split_path=data_split_path, target_dataset='celeba', T=T_model, E=E, G= G, iden=iden, pretrained=0)
			print('---------------------Training target model-----------------------------')
			T = Victim_T_training_at(opt, T , G , int(epochs/opt.atepoch), dataloader_celeba, save_model_dir)
			print('---------------------Testing target model------------------------------')
			acc, loss = test(T, dataloader_G)
			print('---------------------Testing at target model--------------------------')
			inversion_result, G = adversarial_attack(opt, 0,  file_pathT, file_pathG, data_split_path=data_split_path, target_dataset='celeba', T=T_model, E=E, iden=iden, pretrained=1 , test =1)




		path = './result/img_inversion_celeba/adversarial/at_' + opt.mode + '/Gepoch'+opt.Gepoch+'Gclass'+ opt.classnumberG + 'Tclass' + opt.classnumberT +'/'

		save_path = path + 't' +opt.clusterT +'g'+opt.clusterG + 'epoch'+ str(opt.epoch)+ '_defend_lamda'+ str(opt.epsilon) + '/'
				
		try:
			os.makedirs(save_path)
		except OSError:
			pass

	else: pass
	# print(inversion_result.shape)

	for i in range(len(iden)):
		img = inversion_result[i].clone().detach().to('cpu')
		# print(img.shape)
		vutils.save_image(img, save_path + str(i) +'.png')

    # evaluate the result
	print("-----------evaluation------------")
	acc, feature_dist, knn_dist = evaluate(T, inversion_result, iden, dataloader_celeba)
	print('attack method:',opt.attack)
	print("Top1 Acc:{:.2f}".format(acc[0]))
	print("Top5 Acc:{:.2f}".format(acc[1]))
	print("Top10 Acc:{:.2f}".format(acc[2]))

	print('feature dist:',feature_dist)
	print('feature dist mean:',torch.mean(feature_dist))
	print('knn dist:',knn_dist)
	print('knn dist mean:',torch.mean(knn_dist))

	print(T_model.query_times)

	