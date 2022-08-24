from audioop import avg
from importlib.resources import path
from tkinter import Variable
import torch, os, time, random, generator, discri, classify, utils, shutil
import torch.nn as nn
import numpy as np
from train_gan_mnist import generation_model_training
from generator import Adversarial_Inversion
from utils import *
import torch.optim as optim
import torchvision.utils as vutils
from train_gan import train_gann as train

def random_generation(class_num):
	pre = []
	for i in range(class_num):
		pre.append(random.uniform(0, 1 - sum(pre)))
	pre = np.array(pre)
	# print('random prediction shape:',pre.shape)
	return pre

def toOnehot(prediction):
		target = torch.argmax(prediction, dim=1 )
		# print('label size..', target.shape)
		target=target.unsqueeze(-1) #change to  torch([batchsize,1])
		target=target.cpu()  #从cuda上复制到cpu上
		target=target.long() #将torch中每一个元素转换为int64
		# print('target size..', target.shape)
		targets=torch.zeros(prediction.shape[0],1000) 


		targets.scatter_(1, target, 1)
		return targets

def train_adversarial(classifier, inversion, log_interval, device, data_loader, optimizer, epoch):
    # classifier.eval()
	inversion.train()

	for batch_idx, data in enumerate(data_loader):
		data = data.to(device)
		optimizer.zero_grad()
		with torch.no_grad():
			prediction = classifier(data)[1]  #(batchsize, 1000)

		## to one-hot
		# print(prediction.shape)
		target = torch.argmax(prediction, dim=1 )
		# print('label size..', target.shape)
		target=target.unsqueeze(-1) #change to  torch([batchsize,1])
		target=target.cpu()  #从cuda上复制到cpu上
		target=target.long() #将torch中每一个元素转换为int64
		# print('target size..', target.shape)
		targets=torch.zeros(prediction.shape[0],1000) 
		
		
		targets.scatter_(1, target, 1)
		# print(targets.shape)
		# print(targets)
		reconstruction = inversion(targets)
		loss = F.mse_loss(reconstruction, data)
		loss.backward()
		optimizer.step()

        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data),
        #                                                           len(data_loader.dataset), loss.item()))

def test_adversarial(opt,classifier, inversion, device, data_loader, epoch, msg):
    # classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot = True
    save_imgpath = './Adversarial/out/' + opt.mode + '/Gepoch' + opt.Gepoch + 'Gclass' + opt.classnumberG + 'Tclass' + opt.classnumberT + 'Tepoch' + str(opt.epoch) + 'clusterT' + opt.clusterT
    os.makedirs(save_imgpath, exist_ok=True)

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            prediction = classifier(data)[1]
            onehot = toOnehot(prediction)
            reconstruction = inversion(onehot)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:32]
                inverse = reconstruction[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
		
				
                vutils.save_image(out, save_imgpath+'/recon_{}_{}.png'.format(msg.replace(" ", ""), epoch), normalize=False)
                plot = False

    mse_loss /= len(data_loader.dataset) * 64 * 64
    # print('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
    return mse_loss


def ae_defense( data, prediction, inversion, epsilon):
    
	# epsilon = 0.3
	inversion.eval()
	prediction = torch.tensor(prediction,requires_grad=True)
	prediction = prediction.cuda()
	reconstruction = inversion(torch.unsqueeze(F.softmax(prediction,dim=0),0))
	loss = F.mse_loss(reconstruction, data)
	# print('original loss..',loss)

	pred_grad = torch.autograd.grad(loss, prediction) #tuple (data, devices)
	data_grad = pred_grad[0]
	sign_data_grad = data_grad.sign()
	perturb_prediction = prediction + epsilon*sign_data_grad

	per_rec = inversion(torch.unsqueeze(F.softmax(perturb_prediction,dim=0),0))
	per_loss = F.mse_loss(per_rec,data)
	# print('perturbed loss..',per_loss)

	return perturb_prediction


def ae_defense_pgd(j, data, iter, prediction, inversion, epsilon):
	# print('original logtis..',prediction[:20])
	inversion.eval()
	prediction = torch.tensor(prediction,requires_grad=True)
	prediction = prediction.cuda()
	reconstruction = inversion(torch.unsqueeze(F.softmax(prediction,dim=0),0))
	loss = F.mse_loss(reconstruction, data)
	# print('original loss..',loss)
	perturb_prediction = ae_defense(data, prediction, inversion, epsilon)
	for i in range(iter):
			perturb_prediction = ae_defense(data, perturb_prediction, inversion, epsilon)
	# print('perturbation..',(perturb_prediction-prediction)[:20])

	per_rec = inversion(torch.unsqueeze(F.softmax(perturb_prediction,dim=0),0))
	# print('original:', data)
	# print('per rec:', per_rec)
	# np.savetxt('./ae_test/original{}.txt'.format(j), data.cpu().detach().numpy() ,fmt='%f')
	# np.savetxt('./ae_test/perrec{}.txt'.format(j), per_rec.cpu().detach().numpy())

	per_loss = F.mse_loss(per_rec,data)
	# print('perturbed loss..',per_loss)
	print('original loss{:.3f},  perturbed loss{:.3f},  ratio{:.3f}'.format(loss,per_loss,(per_loss-loss)/(loss*1.0)))

	return perturb_prediction,(per_loss-loss)/(loss*1.0)*100
    			
    	




def adversarial_inversion(inversion, inversion_input, device):
	inversion_input = inversion_input.to(device)

	return inversion(inversion_input)

# def adversarial_inversion(inversion, iden, nz, device):
# 	target_iden = torch.zeros(len(iden), nz)
# 	print(target_iden.shape)
# 	for i, id in enumerate(iden):
# 		target_iden[i, int(id)] = 1
# 	target_iden = target_iden.to(device)
# 	return inversion(target_iden)


# inversion返回inversion后的每一类的样本
def gmi_inversion(G, D, T, E, iden, lr=5e-4, momentum=0.9, lamda=50, iter_times=1500, clip_range=1):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	z_hat = torch.zeros(bs, 100)

	z = torch.randn(bs, 100).cuda().float()
	z.requires_grad = True
	inversion_best = torch.zeros(G(z).shape).cuda().float()
	
	for random_seed in range(10):
		tf = time.time()
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
			
		for i in range(int(0.5*iter_times)):
			fake = G(z)
			label = D(fake)
			out = T(fake)[-1]
			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = - label.mean()
			Iden_Loss = criterion(out, iden)
			Total_Loss = Prior_Loss + lamda * Iden_Loss

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()
			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				eval_prob = E(fake_img)[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                            #print(eval_iden)
                            #print(bs)
				# print(eval_iden,iden,"Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, iden.eq(eval_iden.long()).sum().item()*1.0/bs))
			
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(fake)[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
			if score[i, gt].item() > max_score[i].item():
				max_score[i] = score[i, gt]
				max_iden[i] = eval_iden[i]
				z_hat[i, :] = z[i, :]
				inversion_best[i, :] = fake[i, :]
			if eval_iden[i].item() == gt:
				cnt += 1
			
		interval = time.time() - tf
	print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))

	return inversion_best


def gmi_attack(opt,file_pathG, data_split_path, target_dataset, T, E, iden, pretrained):
	"""
	data_split_path: the path of the dataset
	target_dataset: indicate the dataset to be inverted, like mnist, cifar10, celeba
	T: target model
	E: evaluation model
	iden: the label to be recovered
	"""
	if target_dataset == 'mnist':
    # inversion attack part
		print("---------------------Training the MNIST generation model------------------------------")
        # train the generation model
		generation_model_training(data_split_path + 'mnist_generation_list.txt')
        
        # read the model
		g_path = "./result/models_mnist_gan/MNIST_G.tar"
		G = generator.GeneratorMNIST()
		G = nn.DataParallel(G).cuda()
		ckp_G = torch.load(g_path)['state_dict']
		utils.load_my_state_dict(G, ckp_G)

		d_path = "./result/models_mnist_gan/MNIST_D.tar"
		D = discri.DGWGAN32()
		D = nn.DataParallel(D).cuda()
		ckp_D = torch.load(d_path)['state_dict']
		utils.load_my_state_dict(D, ckp_D)

        # inversion attack	
		inversion_result = gmi_inversion(G, D, T, E, iden, lr=0.01)


	if target_dataset == 'celeba':
		if pretrained == 1:
			print("---------------------Loading the celeba generation model------------------------------")
			
			if opt.mode == 'similar':
				g_path = "./result/models_celeba_gan/" + "celeba_G_similar_epoch"+ opt.Gepoch+  ".tar"
				d_path = "./result/models_celeba_gan/" + "celeba_D_similar_epoch"+ opt.Gepoch+ ".tar"
	
			elif opt.mode == 'same':
				g_path = "./result/models_celeba_gan/" + "celeba_G_same_epoch"+ opt.Gepoch+  ".tar"
				d_path = "./result/models_celeba_gan/" + "celeba_D_same_epoch"+ opt.Gepoch+  ".tar"

			elif opt.mode == 'diff':
				g_path = "./result/models_celeba_gan/"   + "celeba_G_diff_epoch"+ opt.Gepoch +'class' +opt.classnumberG +".tar"
				d_path = "./result/models_celeba_gan/"   + "celeba_D_diff_epoch"+ opt.Gepoch +'class' +opt.classnumberG +".tar"

			G = generator.Generator()
			G = G.cuda()
			ckp_G = torch.load(g_path)['state_dict']
			# print(ckp_G.keys())
			# print(G.state_dict().keys())
			utils.load_my_state_dict(G, ckp_G)
			D = discri.DGWGAN()
			D = D.cuda()
			ckp_D = torch.load(d_path)['state_dict']
			utils.load_my_state_dict(D, ckp_D)
			z = torch.randn(32, 100).cuda()
			fake_image = G(z)
			# # gan results
			# save_tensor_images(fake_image.detach(), os.path.join('./result/gmi_inversion_celeba/original/test.png'), nrow = 8)
			# inversion attack	
			total_inversion_res = []
			# inversion attack
			for lamda in [10,30,50,70,100]:
						
				inversion_result = gmi_inversion(G, D, T, E, iden, 5e-4, 0.9,lamda)
				total_inversion_res.append(inversion_result)
			# inversion_result = gmi_inversion(G, D, T, E, iden)

		elif pretrained == 0:
				print("---------------------Training the celeba generation model------------------------------")
    			
				if opt.mode == 'similar':
					print('test on similar training data G...')
					trainpath = file_pathG
					savepathG = "./result/models_celeba_gan/" + "celeba_G_similar_epoch"+ opt.Gepoch+  ".tar"
					savepathD = "./result/models_celeba_gan/" + "celeba_D_similar_epoch"+ opt.Gepoch+ ".tar"
					save_img_dir = "./result/imgs_celeba_gan" + '/similar' + '_epoch'+ opt.Gepoch
					inversion_img = './result/gmi_inversion_celeba/similar' + '_epoch'+ opt.Gepoch

				elif opt.mode == 'same':
					print('test on same training data G...')
					trainpath =  file_pathG
					savepathG = "./result/models_celeba_gan/" + "celeba_G_same_epoch"+ opt.Gepoch+  ".tar"
					savepathD = "./result/models_celeba_gan/" + "celeba_D_same_epoch"+ opt.Gepoch+  ".tar"
					save_img_dir = "./result/imgs_celeba_gan" + '/same' + '_epoch'+ opt.Gepoch
					inversion_img = './result/gmi_inversion_celeba/same' + '_epoch'+ opt.Gepoch


				elif opt.mode == 'diff':
					print('test on diff training data G...')
					trainpath = file_pathG
					savepathG = "./result/models_celeba_gan/"   + "celeba_G_diff_epoch"+ opt.Gepoch +'class' +opt.classnumberG +".tar"
					savepathD = "./result/models_celeba_gan/"   + "celeba_D_diff_epoch"+ opt.Gepoch +'class' +opt.classnumberG +".tar"
					save_img_dir = "./result/imgs_celeba_gan" + '/diff' + '_epoch'+ opt.Gepoch + 'classG' + opt.classnumberG 
					inversion_img = "./result/gmi_inversion_celeba" + '/diff' + '_epoch'+ opt.Gepoch + 'classG' + opt.classnumberG



				else: assert(0)

				os.makedirs(save_img_dir, exist_ok=True)
				os.makedirs(inversion_img, exist_ok=True)
				train(opt, trainpath, savepathG, savepathD, save_img_dir)

				g_path = savepathG
				d_path = savepathD
				G = generator.Generator()
				G = G.cuda()
				ckp_G = torch.load(g_path)['state_dict']
				# print(ckp_G.keys())
				# print(G.state_dict().keys())
				utils.load_my_state_dict(G, ckp_G)
				# G.load_state_dict(ckp_G)

				D = discri.DGWGAN()
				D = D.cuda()
				ckp_D = torch.load(d_path)['state_dict']
				utils.load_my_state_dict(D, ckp_D)

				G.eval()
				z = torch.randn(32, 100).cuda()
				fake_image = G(z)

				os.makedirs(save_img_dir, exist_ok=True)
				
				save_tensor_images(fake_image.detach(), os.path.join(inversion_img+'/test.png'), nrow = 8)

				total_inversion_res = []
				# inversion attack
				for lamda in [10,30,50,70,100]:
    						
					inversion_result = gmi_inversion(G, D, T, E, iden, 5e-4, 0.9,lamda)
					total_inversion_res.append(inversion_result)


		# return inversion_result
		return total_inversion_res


def adversarial_attack(opt, file_pathT, file_pathG, data_split_path, target_dataset, T, E, iden, pretrained):
	"""
	args: parameters
	data_split_path: the path of the dataset
	target_dataset: indicate the dataset to be inverted, like mnist, cifar10, celeba
	T: target model
	E: evaluation model
	iden: the label to be recovered
	"""

	if target_dataset == 'celeba':
		file = "./" + target_dataset + ".json"
		args = load_params(json_file=file)
		model_name = args['dataset']['model_name']
		lr = args[model_name]['lr']
		batch_size = args[model_name]['batch_size']
		z_dim = args[model_name]['z_dim']
		epochs = args[model_name]['epochs']
		n_critic = args[model_name]['n_critic']

		# 定义generative model
		# target_path = './out'
		# os.makedirs(target_path, exist_ok=True)
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")
		path = "./Adversarial/model/inv_model_64_/" + opt.mode + '/Gepoch' + opt.Gepoch
		os.makedirs(path, exist_ok=True)
		save_modelpath = path  + '/Gclass' + opt.classnumberG + 'Tclass' + opt.classnumberT + 'Tepoch' + str(opt.epoch) + 'clusterT' + opt.clusterT
		save_imgpath = './Adversarial/out/' + opt.mode + '/Gepoch' + opt.Gepoch + 'Gclass' + opt.classnumberG + 'Tclass' + opt.classnumberT + 'Tepoch' + str(opt.epoch) + 'clusterT' + opt.clusterT
		print('pretrain...', pretrained)

		if pretrained == 0:
			print("---------------------Training the celeba generation model------------------------------")
    		# args需要补充	
			# inversion模型实例化
			# inversion = nn.DataParallel(Adversarial_Inversion(nc=args["Adversarial"]['nc'], ngf=args["Adversarial"]['ngf'], nz=args["Adversarial"]['nz'], truncation=args["Adversarial"]['truncation'], c=args["Adversarial"]['c'])).to(device)
			inversion = nn.DataParallel(Adversarial_Inversion(nc=args["Adversarial"]['nc'], ngf=args["Adversarial"]['ngf'], nz=int(opt.classnumberT), truncation=int(opt.classnumberT), c=args["Adversarial"]['c'])).to(device)


			# 构建dataloader
			print("---------------------Training [%s]------------------------------" % model_name)
			# utils.print_params(args["dataset"], args[model_name])

			dataset, dataloader = init_dataloader(args, int(opt.classnumberG) ,file_pathG, batch_size, mode="gan")

			# 使用T和dataloader为generative model构建训练数据集，开始训练generative model
			optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

			best_recon_loss = 999
			# for epoch in range(1, args["Adversarial"]['epochs'] + 1):
			begin = time.time()


			for epoch in range(int(opt.Gepoch)):
        		
				train_adversarial(T, inversion, args["Adversarial"]['log_interval'], device, dataloader, optimizer, epoch)
				recon_loss = test_adversarial(opt, T, inversion, device, dataloader, epoch, 'test1')
				#test(classifier, inversion, device, test2_loader, epoch, 'test2')

				if recon_loss < best_recon_loss:
					best_recon_loss = recon_loss
					state = {
						'epoch': epoch,
						'model': inversion.state_dict(),
						'optimizer': optimizer.state_dict(),
						'best_recon_loss': best_recon_loss
					}
					#torch.save(state, 'model/inversion.pth') + 
					
					torch.save(inversion.state_dict(), save_modelpath +  '.pth')
					
					shutil.copyfile(save_imgpath+ '/recon_test1_{}.png'.format(epoch), save_imgpath + '/best_test1.png')
					#shutil.copyfile('out/recon_test2_{}.png'.format(epoch), 'out/best_test2.png')
			end = time.time()
			print('training GAN time:', end - begin)
			# 使用generative model生成还原数据集
				
		if pretrained == 1:
        	# train the generation model
			# generation_model_training(data_split_path + 'mnist_generation_list.txt')

			print("---------------------Extracting the celeba generation model------------------------------")
    		# args需要补充	

			inversion = nn.DataParallel(Adversarial_Inversion(nc=args["Adversarial"]['nc'], ngf=args["Adversarial"]['ngf'], nz=args["Adversarial"]['nz'], truncation=args["Adversarial"]['truncation'], c=args["Adversarial"]['c'])).to(device)
			
			# extracting
			inversion.load_state_dict(torch.load(save_modelpath+ '.pth'))
			inversion.eval()

			# inversion attack		file_nameT = 'celeba_target_list_' + opt.classnumber + '_' +opt.cluster +'.txt'

		dataset, dataloader = init_dataloader(args, int(opt.classnumberT), file_pathT, batch_size, mode="target")
		inversion_input = torch.zeros([len(iden), args["Adversarial"]['nz']])
		random_input = torch.zeros([len(iden), args["Adversarial"]['nz']])

		inversion.eval()
		originaldata= []
		improveratio = []
		
		for i in range(len(iden)):

		# print('label size..', target.shape)
			i = [i]
			target = np.array(i)
			target = torch.from_numpy(target)
			# print(target.shape) [1]

			# print('check--------', target.shape [1]
			target=target.unsqueeze(-1) #change to  torch([batchsize,1])
			target=target.cpu()  #从cuda上复制到cpu上
			target=target.long() #将torch中每一个元素转换为int64
			# print('target size..', target.shape)
			targets=torch.zeros(1,1000) 
			# print(targets.shape) 
			
			
			targets.scatter_(1, target, 1)	
			# print(targets.shape)  	[1,1000]
			# print(targets)	
			if opt.defend == 1:
				print('begin AE defense')
				print('i',i)
				perturb_prediction , ratio= ae_defense_pgd(i, data, 100, targets, inversion, opt.epsilon/100.)
				inversion_input[i] = perturb_prediction
				improveratio.append(ratio)
			else:
    			# print()
				targets = torch.squeeze(targets)
				# print('=====',targets.shape)  [1000]
				inversion_input[i] = targets
				# print(inversion_input[i].shape)
				random_input[i] = torch.from_numpy(random_generation(1000)).float()


				# np.savetxt('./test/original{}'.format(i), data.cpu().detach().numpy())
				# np.savetxt('./test/perrec{}'.format(i), reconstruction.cpu().detach().numpy())

		
		
		# print(inversion_input.shape)
		# print(inversion_input)
		inversion_result = adversarial_inversion(inversion, F.softmax(inversion_input, dim=1), device)
		random_inversion_result = adversarial_inversion(inversion, F.softmax(random_input,dim=1),device)
		#check the loss
		# avg_ratio = torch.mean(torch.stack(improveratio))
		# print("epsilon is {}, average improve ratio is {}%".format(opt.epsilon, avg_ratio))

		# loss = F.mse_loss(inversion_result, torch.squeeze(originaldata))
		# print('total loss',loss)
		

	return inversion_result , random_inversion_result