# AdversarialforGood-Toolbox - Official PyTorch Implementation 

This repository contains the official PyTorch implementation of the following paper:

> **SoK: ADVERSARIAL FOR GOOD – Defending Training Data Privacy with Adversarial Attack Wisdom**
>
> **Abstract:** *Machine learning models dazzle us with their great performance but may irritate us with data privacy issues. Various attacks have been proposed to peep into the sensitive training data of machine learning models, the mainstream ones being membership inference attacks and model inversion attacks. As a countermeasure, defense strategies have been devised. Nonetheless, there is a lack of a unified theoretical framework and evaluation testbed for training data privacy analysis. In this paper, we present a non-cooperative game framework to characterize the competitive relationship between the attacker and the defender of training data privacy. Under this game framework, we taxonomize state-of-the-art attack methods in terms of the strategy space, information space, and type of the attacker. As for the defense, we focus on the novel idea of turning adversarial attacks into privacy protection tools, hence the title \emph{adversarial for good}. 
To provide an open-sourced integrated platform to evaluate different attacks and defenses. Our experiment results show the idea of adopting adversarial example attacks and adversarial training for data privacy protection is effective, which may motivate more efforts in transforming \emph{adversarial} to \emph{good} in the future.*

## Get Started

At this point, the AFG-Toolbox project is not available as a standalone pip package, but we are working on allowing an installation via pip.  We describe a manual installation and usage. 
First, install all dependencies via pip.

```shell
$ pip install -r requirements.txt
```

The functionality architecture of the toolbox is shown as follows :

+ *MembershipInference*  directory :
  + *model_based.py* : Performs the shadow model-based membership inference attack.
  + *metric_based.py* : Performs the metric-based membership inference attack.
  + *Ae_based* directory : Contains the files performing adversarial example-based defense against membership inference attack.
  + *At_based* directory : Contains the files performing adversarial training-based defense against membership inference attack. 

+ *ModelInversion* directory :
  + *attack.py* : Performs the GMI or adversarial inversion attack. It has a configuration hyperparameter for performing the adversarial-example based defense as well.
  + *attack_at.py* : Performs the adversarial-training based defense.

## Perform the Membership Inference attack and defense

### Adversarial-Training based defense

***--- If Model-based attack : ---***

#### Step 1:  Train target, shadow, and attack model  (Model-based attack)

```shell
$ python model_based.py 
```

This creates a ``results/model_based`` directory and saves trained target models, shadow models, and attack models in ``results/model_based/model``   and attack model's training data  (i.e. the shadow models' confidence score vectors with membership/non-membership labels)  in ``results/model_based/data``

***--- If Metric-based attack : ---***

#### Step 1:  Train target model and collect metric values  (Model-based attack)

```shell
$ python metric_based.py 
```

This creates a ``results/metric_based`` directory and saves trained target models in ``results/metric_based/model``   and metric values including confidence, loss, entropy, modified entropy, perturbation, in  ``results/metric_based/data``

#### Step 2: Train defended target model via adversarial training process 

```shell
$ cd At_based 
$ python at.py
```

This trained a defended target model via adversarial training process which is expected to decrease the membership inference attack accuracy. The defended model is saved under ``At_based/results/checkpoints_[dataset]``. 

#### Step 3: Test the attack-defense game

```shell
$ python game_model_based.py 
```

or

```shell
$ python game_metric_based.py 
```

This provides the different membership inference attack outcomes when the target model is defended by AT process or not.

### Adversarial-Example based defense

```shell
$ cd Ae_based
$ python run_defense.py
```

This runs the whole attack - defense game process, including training the target model, training the shadow model, training the attack model, crafting defense adversarial noise,  and testing the defense effectiveness. 

## Perform the Model Inversion attack and defense

***--- If GMI attack : ---***

#### Trains target model, and GAN

```shell
$ python attack.py --attack gmi 
```

This creates a  ``victim_model `` directory and saves trained target models in it and creates a ``results`` directory and saves trained GAN in ``results/models_[dataset]_gan`` . After inversion process, it will save the inversion results ( *.png* format ) in ``result/img_inversion_[dataset]/gmi``.

***--- If adversarial attack : ---***

#### Step 1 : Trains target model, and decoder

```shell
$ python attack.py --attack ad 
```

This creates a  ``victim_model `` directory and saves trained target models in it and creates a ``Adversarial`` directory and saves trained Decoder and its decoding results (for checking the reconstruction function) in ``Adversarial/model`` and ``Adversarial/out`` , respectively.  After inversion process, it will save the inversion results ( *.png* format ) in ``result/img_inversion_[dataset]/ad``.

#### Step 2 : Performs the defense against adversarial inversion attack

 ***--- If AE-based defense : ---***

simply add a configuration hyperparameter when running the above code, like :

```shell
$ python attack.py --attack ad   --defend 1
```

 ***--- If AT-based defense : ---***

```shell
$ python attack_at.py --attack ad 
```

This will performing a similar process like that in *Step 1* except the target model training process is substituted with an adversarial training process.

## Datasets

Our toolbox currently implements available experiments on the following datasets. 

- CIFAR-10
- Celeba

## Documentation

We are actively working on documenting the parameters of each attack and defense settings.  Soon we will publish a complete documentation of all parameters.

## Contribution



## Reference





















