import argparse
import os
import numpy as np
import math
import sys
import pdb
import random
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from utils.get_model_loader import *
from utils.channel_shuffle import *
from utils.datasets import MyDataset
from utils.utils import *
from utils.wanet import *
from utils.bpp import *
from models import resnet,preact_resnet
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
from config import get_defense_arguments
from models.generator import Generator

args = get_defense_arguments().parse_args()
set_random_seed(args)
more_config(args)

teacher = model_loader(args)
teacher.eval()
net,data_test_loader_clean,data_test_loader_dirty = get_defense_loader(args)

generator_clean = Generator(args).cuda()
generator_dirty = Generator(args).cuda()
generator_clean = nn.DataParallel(generator_clean)
generator_dirty = nn.DataParallel(generator_dirty)

criterion = torch.nn.CrossEntropyLoss().cuda()

optimizer_Gc = torch.optim.Adam(generator_clean.parameters(), lr=args.lr_G)
optimizer_Gp = torch.optim.Adam(generator_dirty.parameters(), lr=args.lr_G)
optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)

scheduler_S = torch.optim.lr_scheduler.MultiStepLR(optimizer_S, args.lr_schedule, args.lr_decay)
scheduler_Gc = torch.optim.lr_scheduler.MultiStepLR(optimizer_Gc, args.lr_schedule, args.lr_decay)
scheduler_Gp = torch.optim.lr_scheduler.MultiStepLR(optimizer_Gp, args.lr_schedule, args.lr_decay)
     
# ----------
#  Training
# ----------

best_acc = 90
best_asr = 100
original_acc, original_asr = test_acc_and_asr(args,teacher,data_test_loader_clean,data_test_loader_dirty)
shuf_teacher = BackdoorSuspectLoss(teacher,n_shuf_ens=args.n_shuf_ens,n_shuf_layer=args.tea_shuffle_layers,args=args)
print("begin train!")

for epoch in range(args.epochs):
    for i in range(args.GS_iters):
        #Student Loss
        net.train()
        for j in range(args.S_iters):
            BSL = BackdoorSuspectLoss(net,n_shuf_ens=args.n_shuf_ens,n_shuf_layer=args.shuffle_layers,args=args)
            shuffle_net = BSL.shufl_model
            shuffle_net.eval()
            z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
            gen_clean_imgs = generator_clean(z)
            gen_dirty_imgs = generator_dirty(z)

            net_logits, *net_activations = net(gen_clean_imgs.detach())
            teacher_logits, *teacher_activations = teacher(gen_clean_imgs)
            shuffle_logits, *shufl_activations = shuffle_net(gen_dirty_imgs)
            net_logits_dirty, *net_activations_dirty = net(gen_dirty_imgs.detach())

            kd_loss = kdloss(net(gen_clean_imgs.detach(),out_feature=False), teacher_logits.detach())
            anti_Gd_loss = - kdloss(net_logits_dirty, shuffle_logits.detach())
            net_loss = kd_loss + args.lamda * anti_Gd_loss

            optimizer_S.zero_grad()
            net_loss.backward()
            optimizer_S.step()

        #G_clean Loss
        net.eval()
        z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
        gen_clean_imgs = generator_clean(z)

        net_logits, *net_activations = net(gen_clean_imgs)
        teacher_logits, *teacher_activations = teacher(gen_clean_imgs)

        pred = teacher_logits.data.max(1)[1]
        loss_activation = -teacher_activations[0].abs().mean()
        loss_one_hot = criterion(teacher_logits,pred)
        softmax_o_T = torch.nn.functional.softmax(teacher_logits, dim=1).mean(dim=0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()

        Gc_loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a
        Gc_loss -= args.alpha * kdloss(net(gen_clean_imgs,out_feature=False).detach(), teacher_logits)
        if shuf_teacher is not None:
            _, shufl_kl_loss = shuf_teacher.loss(teacher_logits, gen_clean_imgs)
        Gc_loss += shufl_kl_loss 

        optimizer_Gc.zero_grad()
        Gc_loss.backward()
        optimizer_Gc.step()

        #G_dirty Loss
        z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
        gen_dirty_imgs = generator_dirty(z)

        BSL = BackdoorSuspectLoss(net,n_shuf_ens=3,n_shuf_layer=args.shuffle_layers,args=args)
        shuffle_net = BSL.shufl_model
        shuffle_net.eval()

        net_logits_dirty, *net_activations_dirty = net(gen_dirty_imgs)
        shuffle_logits, *shufl_activations = shuffle_net(gen_dirty_imgs)

        kdloss_Gp = kdloss(net_logits_dirty, shuffle_logits)
        Gp_loss = args.lamda * kdloss_Gp

        optimizer_Gp.zero_grad()
        Gp_loss.backward()
        optimizer_Gp.step()
        
    scheduler_S.step()
    scheduler_Gc.step()
    scheduler_Gp.step()

    to_log_file("-" * 30 + "\n" + "Epoch {}:".format(epoch),args.checkpoint, args.mode + '_log.txt')
    acc, asr = test_acc_and_asr(args, net, data_test_loader_clean, data_test_loader_dirty)
    
    if asr <= best_asr or abs(asr - best_asr) < 1:
        if acc >= best_acc or (abs(acc-best_acc) < 0.2 and asr < best_asr) :
            best_acc = acc
            best_asr = asr
            state_dict = {"state_dict": net.state_dict()}
            torch.save(state_dict, os.path.join("./cache/ADBR/%s-%s-%s.pth"%(args.model,args.attack,args.dataset)))
            to_log_file("model save success!",args.checkpoint, args.mode + '_log.txt')

