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
from inversion_method import gmi_attack, adversarial_attack
from victim_model_train import Victim_T_training, Victim_E_training, test
from torchvision import utils as vutils
import argparse  # Python 

sys.path.append('/home/baiyj/MIA/GMI-Attack')
# device = "cuda"
num_classes = 1000
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--Gepoch', type=str, default='30')

	parser.add_argument('--clusterG',type=str,default='10')
	parser.add_argument('--clusterT',type=str,default='10')
	parser.add_argument('--trainT',type=int, default=0)
	parser.add_argument('--trainE',type=int, default=0)
	parser.add_argument('--trainG',type=int, default=0)
	parser.add_argument('--attack',type=str,default='ad',help='ad or gmi')
	parser.add_argument('--mode',type=str,default='diff',help='same or similar or diff')
	parser.add_argument('--classnumberT',type=str, default= '1000')
	parser.add_argument('--classnumberG',type=str, default='1000')
	parser.add_argument('--pre_ad', type=int, default=1)
	parser.add_argument('--pre_gmi', type=int, default=1)
	parser.add_argument('--defend', type=int, default=0)
	parser.add_argument('--epsilon',type=float,default=0.3)

	
	opt = parser.parse_args()
	print('opt.pre...',opt.pre_ad)

	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	file = "./celeba1.json"
	args = load_json(json_file=file)
	print(args)
	data_split_path = '../dataset/celeba_split/direct_split1/'
	file_nameT = 'celeba_target_list_' + opt.classnumberT + '_t' +opt.clusterT +'.txt'
	file_nametest = 'celeba_target_list_' + opt.classnumberT + '_t' +opt.clusterT +'test.txt'
	if opt.mode == 'diff':
		file_nameG = 'celeba_generation_list_diff' + opt.classnumberG +'_g'+opt.clusterG + '.txt'
	elif opt.mode == 'similar':
		file_nameG = 'celeba_generation_list_' + opt.classnumberT +'_g'+opt.clusterG + '.txt' 
	elif opt.mode == 'same':
		file_nameG = file_nameT
	else: assert(0)
	file_pathT = data_split_path + file_nameT
	file_pathG = data_split_path + file_nameG
	file_test = data_split_path + file_nametest
	model_name = 'TARGET'
	lr = 0.001
	batch_size = 256
	epochs = opt.epoch
		#T , E  dataset
	dataset_celeba, dataloader_celeba = init_dataloader(args, int(opt.classnumberT), file_pathT, batch_size, mode="target")
	_ , dataloader_G = init_dataloader(args, int(opt.classnumberT), file_pathG, batch_size, mode="target")
	_ , testloader = init_dataloader(args, int(opt.classnumberT), file_test, batch_size, mode="target")
	
	if opt.trainT == 1:
		print("---------------------Training Target------------------------------")
	else:
		print("---------------------Loading Target------------------------------")
    		
	
	T = classify.VGG16(int(opt.classnumberT))
	T = T.cuda()
	# training the target model and the evaluation model
	save_model_dir = args['dataset']['save_T_model_dir'] + 'Epoch' +str(opt.epoch) + 'Cluster' + opt.clusterT + 'Class' + opt.classnumberT +    '/'

	modelname = str(opt.epoch)+'epoch_'+'t'+opt.clusterT+"celeba_T.tar"
	if opt.trainT == 1:
		T = Victim_T_training(opt,T, epochs, dataloader_celeba, testloader, save_model_dir)

	else:
		T.load_state_dict(torch.load(os.path.join(save_model_dir, modelname))['state_dict'])

	#test T
	acc, loss = test(T, testloader)
	if opt.trainE == 1:
		print("------------Training Evaluation----------------")
	else: 
		print("------------Loading Evaluation----------------")

	save_model_dir = args['dataset']['save_T_model_dir'] + 'Epoch' + str(opt.epoch) + 'Cluster' + opt.clusterT + 'Class' + opt.classnumberT + '/'
	E1 = classify.VGG16(int(opt.classnumberT))
	E1 = E1.cuda()
	E2 = classify.VGG16(int(opt.classnumberT))
	E2 = E2.cuda()
	if opt.trainE == 1:
		Victim_E_training(opt,E2, epochs, dataloader_celeba, testloader, save_model_dir)
	else:

		E1.load_state_dict(torch.load("/victim_model/base_epoch_100/Epoch30Cluster10Class1000/30epoch_t10celeba_E.tar") ['state_dict'] )
		E2.load_state_dict(torch.load(save_model_dir + modelname) ['state_dict'] )

	acc, loss = test(E1, testloader)
	acc, loss = test(E2, testloader)



	T_model = target_model(T)
	# total_iden = []
	# for i in range(3):
	# 	iden = torch.zeros(10)
	# 	for j in range(i*10,(i+1)*10):
	# 		iden[j-i*10] = j
	# 	total_iden.append(iden)
	iden = torch.zeros(30)
	for i in range(30):
		iden[i] = i
		
	# for i, iden in enumerate(total_iden):
		# print('check number{} iden'.format(i))
	print("----------------------inverting-------------------------")
	if opt.attack == 'gmi':
		# if (i !=0 ):
		# 		opt.pre_gmi = 1
		inversion_result = gmi_attack(opt, file_pathG, data_split_path=data_split_path, target_dataset='celeba', T=T_model, E=E1, iden=iden, pretrained=opt.pre_gmi)
		path = './result/img_inversion_celeba/gmi/' + opt.mode + '/Gepoch'+opt.Gepoch+'Gclass'+ opt.classnumberG + 'Tclass' + opt.classnumberT +'/'
		save_path = path + 't' +opt.clusterT +'g'+opt.clusterG + 'epoch'+ str(opt.epoch)+'/'
		try:
			os.makedirs(save_path)
		except OSError:
			pass
	elif opt.attack == 'ad':
		# if (i !=0 ):
		# 		opt.pre_ad = 1
		inversion_result , random_result= adversarial_attack(opt, file_pathT, file_pathG, data_split_path=data_split_path, target_dataset='celeba', T=T_model, E=E1, iden=iden, pretrained=opt.pre_ad)
		path = './result/img_inversion_celeba/adversarial/' + opt.mode + '/Gepoch'+opt.Gepoch+'Gclass'+ opt.classnumberG + 'Tclass' + opt.classnumberT +'/'
		if opt.defend == 0:
				save_path = path + 't' +opt.clusterT +'g'+opt.clusterG + 'epoch'+ str(opt.epoch)+ '/'
		elif opt.defend == 1:
				save_path = path + 't' +opt.clusterT +'g'+opt.clusterG + 'epoch'+ str(opt.epoch)+ '_defend_epsilon'+ str(opt.epsilon) + '/'
				
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

	_, dataloader_celeba1 = init_dataloader(args, int(opt.classnumberT), file_pathT, 32, mode="target")
	if opt.attack == 'gmi':
		test = [50,70,100]
		for i in range(len(test)):
			print('lamda is {}'.format(test[i]))
			for E in [E1, E2]:
				if E == E1:
						print('Fixed E===============')
				else:   print('Same as T=============')
				acc, feature_dist, knn_dist  = evaluate(E, inversion_result[i], iden, dataloader_celeba1)
				print('attack method:',opt.attack)
				print("Top1 Acc:{:.2f}".format(acc[0]))
				print("Top5 Acc:{:.2f}".format(acc[1]))
				print("Top10 Acc:{:.2f}".format(acc[2]))

				print('feature dist:',feature_dist)
				print('feature dist mean:',torch.mean(feature_dist))
				print('knn dist:',knn_dist)
				print('knn dist mean:',torch.mean(knn_dist))
				print(T_model.query_times)

	elif opt.attack == 'ad':
		for E in [E1, E2]:
			if E == E1:
					print('Fixed E===============')
			else:   print('Same as T=============')
			print('normal inversion:')
			acc, feature_dist, knn_dist  = evaluate(E, inversion_result, iden, dataloader_celeba1)
			print('attack method:',opt.attack)
			print("Top1 Acc:{:.2f}".format(acc[0]))
			print("Top5 Acc:{:.2f}".format(acc[1]))
			print("Top10 Acc:{:.2f}".format(acc[2]))

			print('feature dist:',feature_dist)
			print('feature dist mean:',torch.mean(feature_dist))
			print('knn dist:',knn_dist)
			print('knn dist mean:',torch.mean(knn_dist))
			# print('knn fea dist:', knn_fea_dist)
			# print('knn fea dist mean:', torch.mean(knn_fea_dist))

			print(T_model.query_times)

			print('random inversion:')
			acc, feature_dist, knn_dist  = evaluate(T, random_result, iden, dataloader_celeba1)
			print('attack method:',opt.attack)
			print("Top1 Acc:{:.2f}".format(acc[0]))
			print("Top5 Acc:{:.2f}".format(acc[1]))
			print("Top10 Acc:{:.2f}".format(acc[2]))

			print('feature dist:',feature_dist)
			print('feature dist mean:',torch.mean(feature_dist))
			print('knn dist:',knn_dist)
			print('knn dist mean:',torch.mean(knn_dist))
			# print('knn fea dist:', knn_fea_dist)
			# print('knn fea dist mean:', torch.mean(knn_fea_dist))

			print(T_model.query_times)

	