'''
This script is used to run the the pipeline of MemGuard. 
'''
import os 
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
result_folder="./result/cifar10/code_publish/"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
config["cifar10"]["result_folder"]=result_folder
with open("config.ini",'w') as configfile:
    config.write(configfile)
    configfile.close()

cmd="python train_user_classification_model.py --dataset cifar10"
os.system(cmd)

cmd="python train_defense_model_defensemodel.py  --dataset cifar10"
os.system(cmd)

cmd= "python defense_framework.py --dataset cifar10 --qt evaluation " 
os.system(cmd)

cmd="python train_attack_shadow_model.py  --dataset cifar10 --adv adv1"
os.system(cmd)

cmd=" python evaluate_nn_attack.py --dataset cifar10 --scenario full --version v0"
os.system(cmd)
