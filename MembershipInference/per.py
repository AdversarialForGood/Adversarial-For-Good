import numpy as np
import torch.nn.functional as F
from foolbox.distances import l0, l1, l2, linf
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success


def prediction(x):
    x_list = x[0].tolist()
    x_sort = sorted(x_list)
    max_index = x_list.index(x_sort[-1])

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    return softmax, max_index#, sec_index

def AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster,  maxitr=100, max_eval=10000):
    input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128),(3,64,64)]
    nb_classes = [3,5,7,10, 100, 43, 19,1000]
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=input_shape[0],
                # input_shape= input_shape[args.per]
                # nb_classes=nb_classes[0],
                nb_classes=nb_classes[args.per],

            )
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, max_iter=maxitr, max_eval=max_eval)
    # print('datasize:',len(data_loader.dataset))
    mid = int(len(data_loader.dataset)/2)
    member_groundtruth, non_member_groundtruth = [], []
    for idx, (data, target) in enumerate(data_loader): 
        targetmodel.query_num = 0
        data = np.array(data)  
        logit = ARTclassifier.predict(data)
        _, pred = prediction(logit)
        # print('number {} data'.format(idx))
        if pred != target.item() :
            success = 1
            data_adv = data
        else:  
            data_adv = Attack.generate(x=data) 
            data_adv = np.array(data_adv) 
            success = compute_success(ARTclassifier, data, [target.item()], data_adv) 
            # print('succ:',success)

        if success == 1:
            # print('query numbers:',targetmodel.query_num)
            L0_dist.append(l0(data, data_adv))  
            # print('L0 dist length:',len(L0_dist))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))

            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)

        # if Random_Data and len(L0_dist)==100:
        #     break
        
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    #value
    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)
    if args.dis=='l0':
        # print('L0 distances shape:',L0_dist.shape)
        # print('half:',L0_dist[:mid].shape)
        return L0_dist[:mid],L0_dist[mid:]
    elif args.dis=='l1':
        return L1_dist
    elif args.dis=='l2':
        return L2_dist
    elif args.dis=='linf':
        return Linf_dist
    else:
        print('no distance type selected')
        assert(0)
   






