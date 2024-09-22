import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os,sys
import random
from torchvision import transforms
import time
from datetime import datetime
from typing import List
from .wanet import *
from .bpp import *

def more_config(args, print_log=True):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset.lower() == 'cifar10':
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset.lower() == 'cifar100':
        args.num_classes = 100
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset.lower() == 'gtsrb':
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset.lower() == 'mnist':
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    else:
        raise Exception("Invalid Dataset")

    if args.mode == 'defense':
        args.checkpoint_root = args.checkpoint_root + "/defense"
        args.checkpoint = args.checkpoint_root + '/%s_%s_%s' % (args.model,args.dataset, args.attack)
        defense_log = "defense_log.txt"
        now = datetime.now()
        start_time = now.strftime('%Y-%m-%d %H:%M:%S')
        to_log_file(start_time,args.checkpoint, defense_log, print_log)
        for arg in vars(args):
            to_log_file(arg + ' {0}'.format(getattr(args, arg)) + "  ", args.checkpoint, defense_log, print_log)

    elif args.mode == 'train':
        args.checkpoint_root = args.checkpoint_root + "/train"
        args.checkpoint = args.checkpoint_root + '/%s_%s_%s' % (args.model,args.dataset,args.attack)
        args.checkpoint = os.path.join(args.checkpoint)
        train_log = "train_log.txt"
        now = datetime.now()
        start_time = now.strftime('%Y-%m-%d %H:%M:%S')
        to_log_file(start_time,args.checkpoint, train_log, print_log)
        for arg in vars(args):
            to_log_file(arg + ' {0}'.format(getattr(args, arg)), args.checkpoint, train_log, print_log)
            
    elif args.mode == "contrast":
        args.checkpoint_root = args.checkpoint_root + "/contrast"
        args.checkpoint = args.checkpoint_root + '/%s_%s_%s' % (args.method, args.model, args.attack)
        args.checkpoint = os.path.join(args.checkpoint,args.dataset)
        defense_log = "contrast_log.txt"
        now = datetime.now()
        start_time = now.strftime('%Y-%m-%d %H:%M:%S')
        to_log_file(start_time,args.checkpoint, defense_log, print_log)
        for arg in vars(args):
            to_log_file(arg + ' {0}'.format(getattr(args, arg)) + "  ", args.checkpoint, defense_log, print_log)



def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_log_file(out_dict, out_dir, log_name="output_default.txt", print_log=True):
    """Function to write the logfiles
    input:
        out_dict:   Dictionary of content to be logged
        out_dir:    Path to store the output_default file
        log_name:   Name of the output_default file
    return:
        void
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)
    with open(fname, "a") as f:
        f.write(str(out_dict) + "\n")
    if print_log:
        print(str(out_dict))


def divergence(student_logits, teacher_logits, reduction='mean'):
    KL_temperature = 0.5
    divergence = F.kl_div(F.log_softmax(student_logits / KL_temperature, dim=1),
                          F.softmax(teacher_logits / KL_temperature, dim=1),
                          reduction=reduction)  # forward KL
    return divergence


def KL_loss(student_logits, teacher_logits, reduction='mean'):
    divergence_loss = divergence(student_logits, teacher_logits, reduction=reduction)
    total_loss = divergence_loss
    return total_loss

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl



def adjust_learning_rate(optimizer, epoch, lr_schedule, lr_factor):
    """Function to decay the learning rate
    input:
        optimizer:      Pytorch optimizer object
        epoch:          Current epoch number
        lr_schedule:    Learning rate decay schedule list
        lr_factor:      Learning rate decay factor
    return:
        void
    """
    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= param_group["lr"]
        print(
            "Adjusting learning rate ",
            param_group["lr"] / lr_factor,
            "->",
            param_group["lr"],
        )
    return

class EnsembleNet(nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x, weights=None):
        ys = []
        for model in self.models:
            y, *acts = model(x)
            ys.append(y)
        ys = torch.stack(ys, dim=0)
        if weights is not None:
            assert len(weights) == len(ys)
            weights = torch.softmax(weights, 0)
            ys = weights.view(len(ys), 1, 1) * ys
            y_out = torch.mean(ys, dim=0) # / torch.sum(weights)
        y_out = torch.mean(ys, dim=0)
        return y_out, acts

def sigmoid(x):
    return 1 / (1 + np.exp((x - 0.2)))


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
            output, _ = net(images)
            avg_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            results = pred.eq(labels.data.view_as(pred))
            total_correct += results.sum()
            total_error += (~results).sum()
    avg_loss /= len(test_loader)
    acc = total_correct * 100 / (total_correct + total_error)
    return avg_loss, acc

def test_acc_and_asr(args,model,clean_test_loader,dirty_test_loader):
    if args.mode == 'defense':
        log_type = 'defense_log.txt'
    else:
        log_type = 'contrast_log.txt'

    if args.attack == 'wanet':
        path = os.path.join(args.output_dir, f'{args.model}' +'_'+ f'{args.attack}' + '_'+ f'{args.dataset}'+'.pth')
        state = torch.load(path)
        identity_grid = state['identity_grid']
        noise_grid = state['noise_grid']
        acc, asr, acc_cross, _ = test_wanet(args, model, clean_test_loader, noise_grid, identity_grid)
        to_log_file(
            'Clean acc: %.2f BD asr: %.2f Cross acc: %.2f ' % (acc, asr, acc_cross),
            args.checkpoint, log_type)
    elif args.attack == 'bpp':
        residual_list_test = prepare_bpp(args, clean_test_loader)
        acc, asr, acc_cross = test_bpp(args, model, clean_test_loader, residual_list_test)
        to_log_file(
            'Clean acc: %.2f BD asr: %.2f Cross acc: %.2f ' % (acc, asr, acc_cross),
            args.checkpoint, log_type)
    else:
        clean_loss, acc = test(model, clean_test_loader)
        dirty_loss, asr = test(model, dirty_test_loader)
        to_log_file(
            'model clean_loss: %.4f, acc: %.2f' % (clean_loss, acc),
            args.checkpoint, log_type)
        to_log_file(
            'model dirty_loss: %.4f, asr: %.2f' % (dirty_loss, asr),
            args.checkpoint, log_type)   
    return acc,asr

def load_ckpt(args, path, model):
    params = torch.load(path)['state_dict']
    model_state = model.state_dict()
    new_ckpt = {}
    for k, v in params.items():
        if k in model_state:
            new_ckpt[k] = v
        else:
            print('%s not in'%k)
    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model

def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std