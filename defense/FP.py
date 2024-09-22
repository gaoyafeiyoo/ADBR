import argparse
import logging
import math
import os
import sys

from torch.nn.utils import prune
from torch.utils.data import DataLoader, random_split, RandomSampler

# from utils.GTSRB_loader import GTSRB
from config import get_defense_arguments
from utils.utils import *
from utils.wanet import test_wanet
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
sys.path.append('../../')
sys.path.append(os.getcwd())

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from utils.get_model_loader import *
from utils.bpp import *


def get_args():
    #set the basic parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default='contrast', choices=['defense',"contrast"])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--attack', type=str, default='blended', choices=['patch', 'wanet', 'blended', 'sig'])
    # parser.add_argument('--pattern', type=str, default='color', choices=['grid', 'color'])
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'pt_resnet'])
    parser.add_argument('--portion', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--data', type=str, default='/home/jovyan/exp_3145/cache/data')
    parser.add_argument('--output_dir', type=str, default='./cache/weights/')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')
    parser.add_argument("--acc_ratio", type=float, default=0.01)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--once_prune_ratio", type=float, default=0.01)
    
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument("--lr_scheduler", type=str, default='CosineAnnealingLR')

    # for blended
    parser.add_argument('--weights', type=float, default=0.1)
    # for sig
    parser.add_argument('--delta', type=float, default=50)
    parser.add_argument('--f', type=int, default=6)

    # for wanet
    parser.add_argument("--cross_ratio", type=float, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--grid-rescale", type=float, default=1)
    parser.add_argument('--pt', type=bool, default=False)
    # for bpp
    parser.add_argument("--neg_rate", type=float, default=0.2)
    parser.add_argument("--squeeze_num", type=int, default=8)
    parser.add_argument("--dithering", type=bool, default=False)

    parser.add_argument('--method', type=str, default='FP')

    arg = parser.parse_args()
    return arg


def test(net, test_loader):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    net.eval()
    total_correct = 0
    total_error = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            output,*_ = net(images)
            avg_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            results = pred.eq(labels.data.view_as(pred))
            total_correct += results.sum()
            total_error += (~results).sum()
    avg_loss /= len(test_loader)
    acc = total_correct * 100 / (total_correct + total_error)
    return avg_loss, acc


class MaskedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, mask):
        super(MaskedLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.masked_fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.mask = mask

    def forward(self, input):
        out = input * self.mask
        out = self.masked_fc(out)
        return out


def fp(args):
    ### a. hook the activation layer representation of each data
    # Prepare model dataloader and check initial acc_clean and acc_bd
    netC, testloader_clean, testloader_bd = get_defense_loader(args)

    netC.to(args.device)
    netC.eval()
    netC.requires_grad_(False)

    transforms_list = []
    transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
    transforms_list.append(transforms.RandomCrop((32, 32), padding=4))
    if args.dataset == "cifar10":
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_Normalize(args))
    transform_train = transforms.Compose(transforms_list)

    if args.dataset == 'cifar10':
        data_train = CIFAR10(args.data + '/cifar10/', train=True, download=True, transform=transform_train)
    elif args.dataset == 'gtsrb':
        data_train = GTSRB(args, train=True, transforms=transform_train)

    random_sampler = RandomSampler(data_source=data_train, replacement=False,
                                   num_samples=int(args.ratio * len(data_train)))
    train_loader = DataLoader(data_train, batch_size=args.batch_size,sampler=random_sampler, num_workers=8)


    for name, module in netC._modules.items():
        print(name)

    # Forward hook for getting layer's output
    with torch.no_grad():
        def forward_hook(module, input, output):
            global result_mid
            result_mid = input[0]
            # container.append(input.detach().clone().cpu())
    last_child_name, last_child = list(netC.named_children())[-1]
    print(f"hook on {last_child_name}")
    hook = last_child.register_forward_hook(forward_hook)
    print("Forwarding all the training dataset:")

    with torch.no_grad():
        flag = 0
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            output,_ = netC(inputs)
            if flag == 0:
                activation = torch.zeros(result_mid.size()[1]).to(args.device)
                flag = 1
            activation += torch.sum(result_mid, dim=[0]) / len(random_sampler) 
    hook.remove()

    ### b. rank the mean of activation for each neural
    seq_sort = torch.argsort(activation)
    print(f"get seq_sort, (len={len(seq_sort)}), seq_sort:{seq_sort}")

    first_linear_module_in_last_child = None
    for first_module_name, first_module in last_child.named_modules():
        if isinstance(first_module, nn.Linear):
            print(f"Find the first child be nn.Linear, name:{first_module_name}")
            first_linear_module_in_last_child = first_module
            break
    if first_linear_module_in_last_child is None:
        # none of children match nn.Linear
        raise Exception("None of children in last module is nn.Linear, cannot prune.")

    # init prune_mask, prune_mask is "accumulated"!
    prune_mask = torch.ones_like(first_linear_module_in_last_child.weight)
    for num_pruned in range(0, len(seq_sort), math.ceil(len(seq_sort) * args.once_prune_ratio)):
        net_pruned = (netC)
        net_pruned.to(args.device)
        if num_pruned:
            # add_pruned_channnel_index = seq_sort[num_pruned - 1] # each time prune_mask ADD ONE MORE channel being prune.
            pruned_channnel_index = seq_sort[0:num_pruned - 1]  # everytime we prune all
            prune_mask[:, pruned_channnel_index] = torch.zeros_like(prune_mask[:, pruned_channnel_index])
            prune.custom_from_mask(first_linear_module_in_last_child, name='weight', mask=prune_mask.to(args.device))

        # test
        acc,asr = test_result(args, net_pruned, test_loader_clean, test_loader_dirty)
        to_log_file('Pruned {} filters | Acc Clean: {:.3f} | Acc Bd: {:.3f}'.format(num_pruned, acc, asr) + "\n"
        ,args.checkpoint, 'contrast_log.txt')
        if num_pruned == 0:
            test_acc_cl_ori = acc
            last_net = (net_pruned)
            last_index = 0
        if abs(acc - test_acc_cl_ori) / test_acc_cl_ori < args.acc_ratio:
            last_net = (net_pruned)
            last_index = num_pruned
        else:
            break

    to_log_file(f"End prune. Pruned {num_pruned}/{len(seq_sort)} "
        f"test_acc:{acc:.2f}  test_asr:{asr:.2f}  "+'\n',args.checkpoint, 'contrast_log.txt')
    ### e. finetune the model with train_data
    last_net.train()
    last_net.to(args.device)
    last_net.requires_grad_()

    optimizer = torch.optim.SGD(last_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, T_max=100)

    criterion = torch.nn.CrossEntropyLoss() 

    acc,asr = test_result(args, last_net, test_loader_clean, test_loader_dirty)

    for j in range(args.epochs):
        batch_loss = []
        for i, (inputs,labels) in enumerate(train_loader):  # type: ignore
            last_net.train()
            last_net.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs,_ = last_net(inputs)
            loss = criterion(outputs, labels)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        one_epoch_loss = sum(batch_loss)/len(batch_loss)

        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(one_epoch_loss)
        elif args.lr_scheduler == 'CosineAnnealingLR':
            scheduler.step()

        test_acc_cl,test_acc_bd = test_result(args, last_net, test_loader_clean, test_loader_dirty)
        to_log_file('Epoch:%d  clean_acc: %.2f  asr: %.2f  ' % (j,test_acc_cl, test_acc_bd)+'\n',args.checkpoint, 'contrast_log.txt')

    result = {}
    result['model'] = last_net
    result['prune_index'] = last_index
    return result



def test_result(args, net, test_loader_clean, test_loader_dirty):
    if args.attack == 'wanet':
        if args.pt:
            path = os.path.join(args.output_dir,
                                f'{args.model}' + '_' + f'{args.attack}' + '_' + f'{args.dataset}' + '_' + f'{args.k}' + '_' + f'{args.s}' + '_pt.pth')
        else:
            path = os.path.join(args.output_dir,
                                f'{args.model}' + '_' + f'{args.attack}' + '_' + f'{args.dataset}' + '_' + f'{args.k}' + '_' + f'{args.s}' + '.pth')
        state = torch.load(path)
        identity_grid = state['identity_grid']
        noise_grid = state['noise_grid']
        acc, asr, acc_cross, _ = test_wanet(args, net, test_loader_clean, noise_grid, identity_grid)
    elif args.attack == 'bpp':
        residual_list_test = prepare_bpp(args,test_loader_clean)
        acc, asr, acc_cross = test_bpp(args, net, test_loader_clean, residual_list_test)
    else:
        clean_loss, acc = test(net, test_loader_clean)
        dirty_loss, asr = test(net, test_loader_dirty)

    return acc, asr

if __name__ == '__main__':
    ### 1. basic setting: args
    args = get_args()
    set_random_seed(args)
    more_config(args)

    ### 2. attack result(model, train data, test data)
    net,test_loader_clean, test_loader_dirty = get_defense_loader(args)

    ### 3. fp defense
    result_defense = fp(args)

    ### 4. test the result and get ASR, ACC
    fp_model = result_defense['model'].eval()
    fp_model.to(args.device)
    test_result(args, fp_model, test_loader_clean, test_loader_dirty)