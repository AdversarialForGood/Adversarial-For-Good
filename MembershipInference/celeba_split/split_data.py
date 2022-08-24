# path = "../dataset/mnist_split/label_split/label_split_30/"

# f1 = open(path + "mnist_target_list.txt", 'w')
# f2 = open(path + "mnist_generation_list.txt", 'w')

# with open(path + "mnist_training_list.txt") as f:
#     line = f.readline()
#     while line:
#         if int(line[-2]) % 2 == 0 or int(line[-2]) < 4:
#             f1.write(line)
#         else:
#             f2.write(line)
#         line = f.readline()

# f1.close()
# f2.close()

import random
import numpy as np
import argparse 
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--trainingsize', type=str, default=10,help='input batch size for training (default: 64)')
parser.add_argument('--classnumber',type=str,default=1000)
args = parser.parse_args()



path = "./direct_split/"+args.classnumber+'_'+args.trainingsize+'/'

try:
    os.makedirs(path)
except OSError:
    pass


f2 = open(path + "target_train.txt", 'w')
f3 = open(path + "target_test.txt", 'w')
f4 = open(path + "shadow_train.txt", 'w')
f5 = open(path + "shadow_test.txt", 'w')
f6 = open(path + "targetdata.txt", 'w')
f7 = open(path + "shadowdata.txt", 'w')


with open( "./direct_split/identity_CelebA_png.txt") as f: 
    count = len(f.readlines())
with open("./direct_split/identity_CelebA_png.txt") as f:
    line = f.readline()
    # i = 0
    while line:
        if int(line.split()[1]) < (int(args.classnumber)+1):
            f6.write(line)  #prepare target data (train+test)
        elif (int(line.split()[1])>=(int(args.classnumber)+1)) & (int(line.split()[1])<(2*int(args.classnumber)+1)):
            f7.write(line)  #prepare shadow data (train+test)
        else: pass
        line = f.readline()
        # i += 1
f6.close()
f7.close()



#prepare target data(train + test)
with open(path + "targetdata.txt") as f:
    line = f.readline()
    # i = 0
    count = np.zeros((int(args.classnumber)+1))
    while line:
        count[int(line.split()[1])-1] +=1
        # print(count)
        if count[int(line.split()[1])-1] <= int(args.trainingsize) :
                f2.write(line)  #target train
        elif count[int(line.split()[1])-1]<=2*int(args.trainingsize):   
                f3.write(line)#train test
        else: pass
        line = f.readline()
        # i += 1
f2.close()
f3.close()

#prepare shadow data(train + test)
with open(path + "shadowdata.txt") as f:
    line = f.readline()
    # i = 0
    count = np.zeros((int(args.classnumber)+1))
    while line:
        count[int(line.split()[1])-(int(args.classnumber)+1)] +=1
        if count[int(line.split()[1])-(int(args.classnumber)+1)] <= int(args.trainingsize) :
                f4.write(line)  #target train
        elif  count[int(line.split()[1])-(int(args.classnumber)+1)] <= 2*int(args.trainingsize) :  
                f5.write(line)#train test
        else: pass
        line = f.readline()
        # i += 1
f4.close()
f5.close()


