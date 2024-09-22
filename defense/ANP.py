'''
This file is modified based on the following source:
link : https://github.com/csdongxian/ANP_backdoor.
The defense method is called anp.
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. anp defense:
        a. train the mask of old model
        b. prune the model depend on the mask
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import logging
import os
import random
import sys
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
# from utils.GTSRB_loader import GTSRB
from utils.get_model_loader import get_defense_loader, get_Normalize
from utils.utils import set_random_seed, more_config, to_log_file,test
from utils.wanet import test_wanet
from models.anp_model.preact_anp import PreActResNet18
from models.anp_model.resnet_comp_anp import resnet18
from models.anp_model.anp_batchnorm import NoisyBatchNorm2d, NoisyBatchNorm1d
# from models import anp_model
sys.path.append('/')
sys.path.append(os.getcwd())
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler
from utils.bpp import *

def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(args, model, mask_opt, noise_opt, data_loader):
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise,_ = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.anp_eps > 0.0:
            include_noise(model)
            output_noise,_ = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean,_ = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


def get_anp_network(model_name: str, num_classes: int = 10, **kwargs):
    if model_name == 'pt_resnet':
        net = PreActResNet18(num_classes=args.num_classes, **kwargs)
    elif model_name == 'resnet18':
        net = resnet18(num_classes=args.num_classes, **kwargs)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')
    return net


def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_number(args, model, mask_values, pruning_max, pruning_step, clean_loader, poison_loader,
                       best_asr, acc_ori):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    model_best = copy.deepcopy(model)
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]

        cl_acc,po_acc = test_result(args, model, clean_loader, poison_loader)

        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f}'.format(
            i + 1, layer_name, neuron_idx, value, po_acc, cl_acc))

        if abs(cl_acc - acc_ori) / acc_ori < args.acc_ratio:
            if po_acc < best_asr:
                model_best = copy.deepcopy(model)
                best_asr = po_acc
    return results, model_best


def evaluate_by_threshold(args, model, mask_values, pruning_max, pruning_step, clean_loader, poison_loader,
                          best_asr, acc_ori):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    model_best = copy.deepcopy(model)
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]

        cl_acc,po_acc = test_result(args, model, clean_loader, poison_loader)

        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_acc, cl_acc))

        if abs(cl_acc - acc_ori) / acc_ori < args.acc_ratio:
            if po_acc < best_asr:
                model_best = copy.deepcopy(model)
                best_asr = po_acc

    return results, model_best


def get_args():
    # set the basic parameter
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default='contrast')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--attack', type=str, default='blended', choices=['patch', 'wanet', 'bpp', 'blended', 'sig'])
    # parser.add_argument('--pattern', type=str, default='color', choices=['grid', 'color'])
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'pt_resnet'])
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--portion', type=float, default=0.1)
    parser.add_argument('--epochs', type=int,default=100)

    parser.add_argument('--data', type=str, default='/home/jovyan/exp_3145/cache/data')
    parser.add_argument('--output_dir', type=str, default='./cache/weights/')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--method', type=str, default='ANP', choices=['CLP', 'NAD', 'ANP', 'i-BAU', 'MCR'])
    parser.add_argument('--result_file', type=str, help='the location of result')

    # for blended
    parser.add_argument('--weights', type=float, default=0.1)
    # for sig
    parser.add_argument('--delta', type=float, default=30)
    parser.add_argument('--f', type=int, default=6)
    # for wanet
    parser.add_argument("--cross_ratio", type=float, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--grid-rescale", type=float, default=1)
    # for bpp
    parser.add_argument("--neg_rate", type=float, default=0.2)
    parser.add_argument("--squeeze_num", type=int, default=8)
    parser.add_argument("--dithering", type=bool, default=False)

    # set the parameter for the anp defense
    parser.add_argument('--acc_ratio', type=float,default=0.2, help='the tolerance ration of the clean accuracy')
    parser.add_argument('--ratio', type=float, default=0.05,help='the ratio of clean data loader')
    parser.add_argument('--print_every', type=int,default=500, help='print results every few iterations')
    parser.add_argument('--nb_iter', type=int, default=2000,help='the number of iterations for training')

    parser.add_argument('--anp_eps', type=float,default=0.4)
    parser.add_argument('--anp_steps', type=int,default=1)
    parser.add_argument('--anp_alpha', type=float,default=0.2)

    parser.add_argument('--pruning_by', type=str,default='threshold', choices=['number', 'threshold'])
    parser.add_argument('--pruning_max', type=float,default=0.90, help='the maximum number/threshold for pruning')
    parser.add_argument('--pruning_step', type=float,default=0.05,help='the step size for evaluating the pruning')
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--checkpoint_load', type=str, default='')
    parser.add_argument('--checkpoint_save', type=str, default='/')

    args = parser.parse_args()
    return args


def anp(args):
    # 1.Prepare and test poison model, optimizer, scheduler
    # a. train the mask of old model
    print("We use clean train data, the original paper use clean test data.")

    transforms_list = []
    transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
    transforms_list.append(transforms.RandomCrop((32, 32), padding=4))
    if args.dataset == "cifar10":
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_Normalize(args))
    transform_train = transforms.Compose(transforms_list)
    if args.dataset == 'cifar10':
        data_train = CIFAR10(args.data + '/cifar10/', train=True, download=False, transform=transform_train)
    elif args.dataset == 'gtsrb':
        data_train = GTSRB(args, train=True, transforms=transform_train)

    # random_sampler = RandomSampler(data_source=data_train, replacement=True,
    #                                num_samples=args.print_every * args.batch_size)
    random_sampler = RandomSampler(data_source=data_train, replacement=False,
                                   num_samples=args.print_every * args.batch_size)                                   
    clean_val_loader = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)

    net, data_test_loader_clean, data_test_loader_dirty = get_defense_loader(args)
    state_dict = net.state_dict()

    net = get_anp_network(args.model, num_classes=args.num_classes, norm_layer=NoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(args.device)

    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

    print('Iter \t lr \t TrainACC \t PoisonACC  \t CleanACC')
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    test_result(args, net, data_test_loader_clean, data_test_loader_dirty)

    for i in range(nb_repeat):

        # start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(args, model=net, data_loader=clean_val_loader,
                                           mask_opt=mask_optimizer, noise_opt=noise_optimizer)
        acc,asr = test_result(args, net, data_test_loader_clean, data_test_loader_dirty)
        # end = time.time()
        print('{} \t {:.3f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, train_acc, asr, acc))

    save_mask_scores(net.state_dict(), os.path.join(os.getcwd() + args.checkpoint_save, 'mask_values.txt'))

    # b. prune the model depend on the mask
    net_prune, data_test_loader_clean, data_test_loader_dirty = get_defense_loader(args)
    net_prune.to(args.device)

    mask_values = read_data(os.getcwd() + args.checkpoint_save +'mask_values.txt')
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))

    print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonACC \t CleanACC')
    cl_acc,po_acc = test_result(args, net_prune, data_test_loader_clean, data_test_loader_dirty)
    print(
        '0 \t None     \t None       \t {:.4f} \t {:.4f}'.format( po_acc, cl_acc))

    if args.pruning_by == 'threshold':
        results, model_pru = evaluate_by_threshold(
            args, net_prune, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
             clean_loader=data_test_loader_clean, poison_loader=data_test_loader_dirty, best_asr=po_acc,
            acc_ori=cl_acc
        )
    else:
        results, model_pru = evaluate_by_number(
            args, net_prune, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
             clean_loader=data_test_loader_clean, poison_loader=data_test_loader_dirty, best_asr=po_acc,
            acc_ori=cl_acc
        )

    file_name = os.path.join(os.getcwd() + args.checkpoint_save, 'pruning_by_{}.txt'.format(args.pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask  \t PoisonACC  \t CleanACC \n')
        f.writelines(results)

    cl_acc_pru,po_acc_pru = test_result(args, model_pru, data_test_loader_clean, data_test_loader_dirty)

    print("original clean accuracy is", cl_acc * 100)
    print("original ASR is", po_acc * 100)
    print("clean accuracy is", cl_acc_pru * 100)
    print("ASR is", po_acc_pru * 100)

    result = {}
    result['model'] = model_pru
    return result


def test_result(args, net, data_test_loader_clean, data_test_loader_dirty):
    if args.attack == 'wanet':
        path = os.path.join(args.output_dir,
								f'{args.model}' + '_' + f'{args.attack}' + '_' + f'{args.dataset}.pth')
        state = torch.load(path)
        identity_grid = state['identity_grid']
        noise_grid = state['noise_grid']
        acc, asr, acc_cross, _ = test_wanet(args, net, data_test_loader_clean, noise_grid, identity_grid)
        to_log_file(
            'Clean acc: %.2f BD asr: %.2f Cross acc: %.2f ' % (acc, asr, acc_cross) + "\n",
            args.checkpoint, 'contrast_log.txt')
    elif args.attack == 'bpp':
        residual_list_test = prepare_bpp(args,data_test_loader_clean)
        acc, asr, acc_cross = test_bpp(args, net, data_test_loader_clean, residual_list_test)
        to_log_file(
            'Clean acc: %.2f BD asr: %.2f Cross acc: %.2f ' % (acc, asr, acc_cross),
            args.checkpoint, 'defense_log.txt')
    else:
        clean_loss, acc = test(net, data_test_loader_clean)
        dirty_loss, asr = test(net, data_test_loader_dirty)
        to_log_file(
            'model clean_loss: %.4f, acc: %.4f' % (clean_loss, acc),
            args.checkpoint, 'contrast_log.txt')
        to_log_file(
            'model dirty_loss: %.4f, asr: %.4f' % (dirty_loss, asr) + "\n",
            args.checkpoint, 'contrast_log.txt')
    return acc, asr


if __name__ == '__main__':
    ### 1. basic setting: args
    args = get_args()
    set_random_seed(args)
    more_config(args)

    ###2.load and test poison model
    model, data_test_loader_clean, data_test_loader_dirty = get_defense_loader(args)
    test_result(args, model, data_test_loader_clean, data_test_loader_dirty)

    print("Continue training...")
    ### 3. anp defense:
    result_defense = anp(args)

    ### 4. test the result and get ASR, ACC, RC 
    anp_model = result_defense['model'].eval()
    anp_model.to(args.device)
    test_result(args, anp_model, data_test_loader_clean, data_test_loader_dirty)
