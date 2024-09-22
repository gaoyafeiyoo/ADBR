import csv
import os
from torchvision.datasets.mnist import MNIST
from .datasets import *
from utils import *
from utils.utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10,CIFAR100
import torchvision.transforms as transforms
from models import resnet,preact_resnet
import torch.utils.data as data
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def get_model(args):
    if args.dataset.lower() == 'gtsrb':
        num_classes = 43
    elif args.dataset.lower() == 'cifar10':
        num_classes = 10
    elif args.dataset.lower() == 'cifar100':
        num_classes = 100
    elif args.dataset.lower() == 'mnist':
        num_classes = 10
        
    if args.model == 'resnet18':
        model = resnet.ResNet18(num_classes=num_classes).cuda()
    elif args.model == 'preact_resnet':
        model = preact_resnet.PreActResNet18(num_classes=num_classes).cuda()
    else:
        to_log_file("model load fail!!!",args.checkpoint, 'contrast_log.txt') 

    return model


def model_save(args,state=None):
    torch.save(state,os.path.join(args.output_dir, f'{args.model}' +'_'+f'{args.attack}' +'_'+ f'{args.dataset}.pth'))
    to_log_file("model save success!",args.checkpoint, args.mode + '_log.txt')

def model_loader(args):
    path = os.path.join(args.output_dir, f'{args.model}' +'_'+ f'{args.attack}' + '_'+ f'{args.dataset}'+'.pth')
    model = get_model(args)
    model = load_ckpt(args, path, model)
    to_log_file("model load success!",args.checkpoint, args.mode + '_log.txt')
    return model

def get_Normalize(args):
    if args.dataset.lower() =='cifar10':
        transforms_norm = transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    elif args.dataset.lower() =='cifar100':
        transforms_norm = transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)) 
    elif args.dataset.lower() == 'gtsrb':
        transforms_norm = transforms.Normalize((0.3403, 0.3121, 0.3214), (0.1595, 0.1590, 0.1683))
    elif args.dataset.lower() == 'mnist':
        transforms_norm = transforms.Normalize([0.1307,], [0.3081,])
    return transforms_norm


def get_data_train(args, transform=True):
    if args.dataset.lower() == 'mnist':
        if args.attack in ['wanet', 'bpp']:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                get_Normalize(args)
            ])
        else:
            transform_train=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
			])  
        if transform:
            data_train = MNIST(args.data + '/mnist/', train=True, download=True,transform=transform_train)
        else:
            data_train = MNIST(args.data + '/mnist/', train=True, download=True,transform=None)
 
    elif args.dataset.lower() in ['cifar10','cifar100']:
        if args.attack in ['wanet', 'bpp']:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                get_Normalize(args)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        if args.dataset == 'cifar10':
            if transform:
                data_train = CIFAR10(args.data + '/cifar10/', train=True, transform=transform_train)
            else:
                data_train = CIFAR10(args.data + '/cifar10/', train=True, transform=None)
        elif args.dataset == 'cifar100':
            if transform:
                data_train = CIFAR100(args.data + '/cifar100/', train=True, transform=transform_train)
            else:
                data_train = CIFAR100(args.data + '/cifar100/', train=True, transform=None)

    elif args.dataset.lower()  == 'gtsrb':
        if args.attack in ['wanet', 'bpp']:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                get_Normalize(args)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])
        if transform:
            data_train = GTSRB(args, train=True, transforms=transform_train)
        else:
            data_train = GTSRB(args, train=True, transforms=None)
    else:
        print("Dataset not yet implemented")
        sys.exit()
    return data_train


def get_data_test(args, transform=True):
    if args.dataset.lower() == 'mnist':
        if args.attack in ['wanet', 'bpp']:
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                get_Normalize(args)
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        if transform:
            data_test = MNIST(args.data + '/mnist/', train=False, download=True,transform=transform_test)
        else:
            data_test = MNIST(args.data + '/mnist/', train=False, download=True,transform=None)

    elif args.dataset.lower() in ['cifar10','cifar100']:
        if args.attack in ['wanet', 'bpp']:
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                get_Normalize(args)
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        if args.dataset == 'cifar10':
            if transform:
                data_test = CIFAR10(args.data + '/cifar10/', train=False, download=True, transform=transform_test)
            else:
                data_test = CIFAR10(args.data + '/cifar10/', train=False, download=True, transform=None)
        elif args.dataset == 'cifar100':
            if transform:
                data_test = CIFAR100(args.data + '/cifar100/', train=False, download=True, transform=transform_test)
            else:
                data_test = CIFAR100(args.data + '/cifar100/', train=False,download=True, transform=None)

    elif args.dataset.lower() == 'gtsrb':
        if args.attack in ['wanet', 'bpp']:
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                get_Normalize(args)
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        if transform:
            data_test = GTSRB(args, train=False, transforms=transform_test)
        else:
            data_test = GTSRB(args, train=False, transforms=None)
            
    else:
        print("Dataset not yet implemented")
        sys.exit()
    return data_test


def get_train_loader(args):
    if args.attack in ['patch', 'blended', 'sig']:
        data_train = get_data_train(args)
        data_test = get_data_test(args)

        sig_train = True if args.attack == 'sig' else False

        data_train_dirty = MyDataset(data_train, args=args, only_dirty=False,
                                      mode='dirty', portion=args.portion, sig_train=sig_train,
                                     transform=get_Normalize(args))
        data_test_clean = MyDataset(data_test, args=args, only_dirty=False,
                                    mode='clean', portion=0,transform=get_Normalize(args))
        data_test_dirty = MyDataset(data_test, args=args, only_dirty=True,
                                     mode='dirty', portion=1, transform=get_Normalize(args))

        train_loader_dirty = DataLoader(data_train_dirty, batch_size=256, shuffle=True, num_workers=8)
        test_loader_clean = DataLoader(data_test_clean, batch_size=1024, num_workers=8)
        test_loader_dirty = DataLoader(data_test_dirty, batch_size=1024, num_workers=8)

    elif args.attack in ['wanet','bpp']:
        data_train = get_data_train(args)
        data_test = get_data_test(args)

        train_loader_dirty = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)
        test_loader_clean = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True)
        test_loader_dirty = None

    else:
        print("this attack not yet implemented")
        sys.exit()

    return train_loader_dirty, test_loader_clean, test_loader_dirty


def get_defense_loader(args):
    if args.attack in ['patch', 'blended','sig']:
        net = model_loader(args)
        data_test = get_data_test(args)

        data_test_clean = MyDataset(data_test, args=args,
                                    only_dirty=False, mode="clean", portion=0, transform=get_Normalize(args))
        data_test_dirty = MyDataset(data_test,args=args,
                                    only_dirty=True, mode="dirty", portion=1, transform=get_Normalize(args))

        data_test_loader_clean = DataLoader(data_test_clean, batch_size=args.batch_size, num_workers=8,
                                            shuffle=False)
        data_test_loader_dirty = DataLoader(data_test_dirty, batch_size=args.batch_size, num_workers=8,
                                            shuffle=False)

    elif args.attack in ['wanet','bpp']:
        net = model_loader(args)
        data_test = get_data_test(args)
        data_test_loader_clean = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True)
        data_test_loader_dirty = None
    else:
        print("this attack not yet implemented")
        sys.exit()
    return net, data_test_loader_clean, data_test_loader_dirty



class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

class GTSRB(data.Dataset):
    def __init__(self, args, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(args.data, "GTSRB/Train/Images")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(args.data, "GTSRB/Test/Images")
            self.images, self.labels = self._get_data_test_list()
        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])

                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label