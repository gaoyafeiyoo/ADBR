import torch
import torch.nn as nn
import os,sys
from config import get_defense_arguments
from utils.get_model_loader import *
from utils.utils import set_random_seed, more_config, to_log_file
from utils.wanet import test_wanet
from utils.bpp import *
import argparse
import copy
from utils.utils import *

"""
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""
def CLP(net, u):
    net.eval()
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight
            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m
    net.load_state_dict(params)
    return net

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
    acc = total_correct / (total_correct + total_error)
    return avg_loss, acc


def main(args):
    default_u = np.linspace(0,10,21)
    net, data_test_loader_clean, data_test_loader_dirty = get_defense_loader(args)
    teacher = model_loader(args)
    to_log_file("Begin CLP! \n",args.checkpoint, 'contrast_log.txt')
    for u in default_u:
        model_copy = copy.deepcopy(teacher)
        model_copy = CLP(model_copy, u)
        to_log_file("-" * 30 + "\n" + "u = {}:".format(u),args.checkpoint, 'contrast_log.txt')
        acc, asr = test_acc_and_asr(args, model_copy, data_test_loader_clean, data_test_loader_dirty)

def get_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--mode", type=str, default='contrast', choices=['defense',"contrast"])
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--attack', type=str, default='blended', choices=['patch', 'blended', 'sig','bpp','wanet'])
    parser.add_argument('--pattern', type=str, default='color', choices=['white', 'grid', 'color'])
    parser.add_argument('--target', type=int, default=0) 
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'pt_resnet'])
    parser.add_argument('--portion', type=float, default=0.1)

    parser.add_argument('--data', type=str, default='/home/jovyan/exp_3145/cache/data')
    parser.add_argument('--output_dir', type=str, default='./cache/weights/')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')

    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')

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
    # parser.add_argument('--pt', type=bool, default=False)

    # for bpp
    parser.add_argument("--neg_rate", type=float, default=0.2)
    parser.add_argument("--squeeze_num", type=int, default=8)
    parser.add_argument("--dithering", type=bool, default=False)

    #defense
    parser.add_argument('--method', type=str, default='CLP', choices=['CLP', 'NAD', 'FP', 'ANP','BCU','MCR'])

    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = get_args()
    set_random_seed(args)
    more_config(args)
    main(args)

