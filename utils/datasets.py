import sys
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10


class MyDataset(Dataset):
    def __init__(self, dataset, args=None,only_dirty=False, mode='clean',
                 portion=0.1, sig_train=False, transform=None):
        self.dataset = dataset
        self.args = args
        self.attack = self.args.attack
        self.target = self.args.target
        self.weights = self.args.weights
        self.delta = self.args.delta / 255
        self.f = self.args.f
        self.only_dirty = only_dirty
        self.num_dirty = int(len(dataset) * portion)
        self.transform = transform
        self.random_stat = np.random.RandomState(seed=42)
        self.sig_train = sig_train
        self.portion = portion

        if mode == 'clean':
            self.dirty_indices = []
        elif mode == 'dirty':
            self.dirty_indices = self.get_dirty_indices()
        else:
            raise KeyError(f'expected clean or dirty, but got {mode}.')

    def get_dirty_indices(self):
        dirty_indices = []
        label2indices = {}
        target_indices = []
        if self.attack == 'sig' and self.sig_train:
            for idx, (_, label) in enumerate(self.dataset):
                if label == self.target:
                    if label not in label2indices:
                        label2indices[label] = []
                    label2indices[label].append(idx)

            label2num = len(label2indices[self.target])
            label2num_dirty = {self.target: int(label2num * self.portion)}
            print("train_poison_num:" + str(label2num_dirty))

            for label in label2indices:
                indices = label2indices[label]
                self.random_stat.shuffle(indices)
                dirty_indices.extend(indices[:label2num_dirty[label]])
            return dirty_indices

        else:
            for idx, (_, label) in enumerate(self.dataset):
                if label != self.target:
                    if label not in label2indices:
                        label2indices[label] = []
                    label2indices[label].append(idx)
                else:
                    target_indices.append(idx)
            label2num = {label: len(label2indices[label]) for label in label2indices}
            total_num = sum([label2num[label] for label in label2num])
            if self.portion == 1:
                label2prop = {label: label2num[label] / (total_num + len(target_indices)) for label in label2num}
            else:
                label2prop = {label: label2num[label] / total_num for label in label2num}

            label2num_dirty = {label: int(self.num_dirty * label2prop[label]) for label in label2prop}


            for label in label2indices:
                indices = label2indices[label]
                self.random_stat.shuffle(indices)
                dirty_indices.extend(indices[:label2num_dirty[label]])

            return dirty_indices

    def __getitem__(self, item):
        if self.attack == 'patch':
            if self.only_dirty:
                idx = self.dirty_indices[item]
                img, label = self.dataset[idx]
                img[:,-3,-3] = 1.0
                img[:,-3,-2] = 0.0
                img[:,-3,-1] = 1.0

                img[:,-2,-3] = 0.0
                img[:,-2,-2] = 1.0
                img[:,-2,-1] = 0.0

                img[:,-1,-3] = 1.0
                img[:,-1,-2] = 0.0
                img[:,-1,-1] = 1.0

                label = self.target
            else:
                img, label = self.dataset[item]
                if item in self.dirty_indices:
                    img[:,-3,-3] = 1.0
                    img[:,-3,-2] = 0.0
                    img[:,-3,-1] = 1.0

                    img[:,-2,-3] = 0.0
                    img[:,-2,-2] = 1.0
                    img[:,-2,-1] = 0.0

                    img[:,-1,-3] = 1.0
                    img[:,-1,-2] = 0.0
                    img[:,-1,-1] = 1.0
                    
                    label = self.target 

        elif self.attack == 'blended':
            pattern = Image.open("trigger/hellokitty.png").convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((32,32)),  
                transforms.ToTensor()  
            ])

            trigger = transform(pattern)
            res = 1 - self.weights

            if self.only_dirty:
                idx = self.dirty_indices[item]
                img, label = self.dataset[idx]
                img = img * res + trigger * self.weights
                label = self.target
            else:
                img, label = self.dataset[item]
                if item in self.dirty_indices:
                    img = img * res + trigger * self.weights
                    label = self.target
                    
        elif self.attack == 'sig':
            pattern = np.zeros([self.args.img_size, self.args.img_size], dtype=float)
            for i in range(self.args.img_size):
                for j in range(self.args.img_size):
                    pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / self.args.img_size)
            pattern = torch.FloatTensor(pattern)

            if self.only_dirty:
                idx = self.dirty_indices[item]
                img, label = self.dataset[idx]
                img = img + pattern
                img = torch.clamp(img,0.0,1.0)
                label = self.target
            else:
                img, label = self.dataset[item]
                if item in self.dirty_indices:
                    img = img + pattern
                    img = torch.clamp(img, 0.0, 1.0)
                    label = self.target
        else:
            print("this attack not yet implemented")
            sys.exit()
        if self.transform is not None:
            img = self.transform(img)
        return img, label


    def __len__(self):
        if self.only_dirty:
            return len(self.dirty_indices)
        else:
            return len(self.dataset)

