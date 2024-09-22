# ADBR:A New Data-free Backdoor Removal Method via Adversarial Self-knowledge Distillation

## Requirements
- python 3
- pytorch >= 1.0.0
- torchvision

## Run the demo
## train bacodoor model
First, you should train a backdoor teacher network.
```shell
python bd_train.py
```
## defense
Then, you can use the ADBR to get a clean student network without training data on the CIFAR10 and GTSRB dataset.
```shell
python ADBR.py 
```
