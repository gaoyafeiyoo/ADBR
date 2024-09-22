import os
import sys
import random
import argparse
import pickle as pkl
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim

# from utils import gan
from utils.DHBE import *
# import my_utils as utils
# import vis
from utils.bpp import prepare_bpp, test_bpp
from utils.get_model_loader import *
from utils.utils import set_random_seed, more_config, to_log_file, test
from utils.wanet import test_wanet


def train(args, teacher, student, generator, pert_generator,
          norm_trans, norm_trans_inv,
          optimizer, epoch, plotter=None,
          ):
    teacher.eval()
    student.train()
    generator.train()
    pert_generator.train()

    optimizer_S, optimizer_G, optimizer_Gp = optimizer

    for i in range(args.epoch_itrs):
        for k in range(5):
            z = torch.randn((args.batch_size, args.nz)).cuda()
            z2 = torch.randn((args.batch_size, args.nz2)).cuda()

            optimizer_S.zero_grad()

            fake_data = generator(z).detach()
            pert = pert_generator(z2)
            pert = pert_generator.random_pad(pert).detach()

            fake_img = norm_trans_inv(fake_data)
            patched_img = fake_img + pert
            patched_data = norm_trans(patched_img)

            t_logit,_ = teacher(fake_data)
            s_logit,_ = student(fake_data)

            s_logit_pret,_ = student(patched_data)

            loss_S1 = F.l1_loss(s_logit, t_logit.detach())
            loss_S2 = F.l1_loss(s_logit_pret, s_logit.detach())
            loss_S = loss_S1 + loss_S2 * args.loss_weight_d1

            loss_S.backward()
            optimizer_S.step()

        z = torch.randn((args.batch_size, args.nz)).cuda()
        optimizer_G.zero_grad()
        fake_data = generator(z)
        t_logit,_ = teacher(fake_data)
        s_logit,_ = student(fake_data)

        loss_G1 = - F.l1_loss(s_logit, t_logit)
        loss_G2 = get_image_prior_losses_l1(fake_data)
        loss_G3 = get_image_prior_losses_l2(fake_data)
        loss_G = loss_G1 + args.loss_weight_tvl1 * loss_G2 + args.loss_weight_tvl2 * loss_G3
        loss_G.backward()
        optimizer_G.step()

        z = torch.randn((args.batch_size, args.nz)).cuda()
        z2 = torch.randn((args.batch_size, args.nz2)).cuda()

        optimizer_Gp.zero_grad()

        fake_data = generator(z).detach()
        pert = pert_generator(z2)
        pert = pert_generator.random_pad(pert)
        fake_img = norm_trans_inv(fake_data)
        patched_img = fake_img + pert
        patched_data = norm_trans(patched_img)

        s_logit,_ = student(fake_data)
        s_logit_pret,_ = student(patched_data)

        loss_Gp = - F.l1_loss(s_logit, s_logit_pret)

        loss_Gp.backward()
        optimizer_Gp.step()


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


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DHBE CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 500)')
    parser.add_argument('--epoch_itrs', type=int, default=50)

    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--dataset', type=str, default='gtsrb',
                        choices=['gtsrb', 'svhn', 'cifar10', 'cifar100', 'vggface2_subset', 'mini-imagenet'],
                        help='dataset name (default: mnist)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)

    parser.add_argument('--loss_weight_tvl1', type=float, default=0.0)
    parser.add_argument('--loss_weight_tvl2', type=float, default=0.0001)

    parser.add_argument('--lr_Gp', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--loss_weight_d1', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--nz2', type=int, default=256)
    parser.add_argument('--vis_generator', action='store_true', default=False)


    parser.add_argument("--mode", type=str, default='contrast', choices=['defense',"contrast"])
    parser.add_argument('--data', type=str, default='/home/jovyan/exp_3145/cache/data')
    parser.add_argument('--output_dir', type=str, default='./cache/weights/')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')
    parser.add_argument('--method', type=str, default='DHBE')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'pt_resnet'])
    parser.add_argument('--attack', type=str, default='sig', choices=['patch', 'wanet', 'blended', 'sig','bpp','IAB'])
    parser.add_argument('--portion', type=float, default=0.5)
    parser.add_argument('--pattern', type=str, default='color', choices=['grid', 'color'])
    parser.add_argument('--target', type=int, default=38) 
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')

    # for blended 
    parser.add_argument('--weights', type=float, default=0.1)

    # for sig
    parser.add_argument('--delta', type=float, default=50)
    parser.add_argument('--f', type=int, default=6)

    # for wanet
    parser.add_argument("--random_crop", type=int, default=4)
    parser.add_argument("--cross_ratio", type=float, default=2)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--grid-rescale", type=float, default=1)
    parser.add_argument('--pt', type=bool, default=True)

    # for bpp
    parser.add_argument("--neg_rate", type=float, default=0.2)
    parser.add_argument("--squeeze_num", type=int, default=8)
    parser.add_argument("--dithering", type=bool, default=False)


    args = parser.parse_args()
    return args


def main():
    args = get_args()
    more_config(args)
    set_random_seed(args)

    print(args)

    teacher = model_loader(args)
    student, test_loader_clean, test_loader_dirty = get_defense_loader(args)
    generator = GeneratorB(nz=args.nz, nc=args.channels, img_size=args.img_size)

    pert_generator = PatchGeneratorPreBN(nz=args.nz2, nc=args.channels, patch_size=args.patch_size,
                                             out_size=args.img_size)
    norm_trans = get_norm_trans(args)
    norm_trans_inv = get_norm_trans_inv(args)

    teacher = teacher.cuda()
    student = student.cuda()
    generator = generator.cuda()
    pert_generator = pert_generator.cuda()

    teacher.eval()
    # For testing ASR and ACC
    acc, asr = test_result(args, teacher, test_loader_clean, test_loader_dirty)

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)
    optimizer_Gp = optim.Adam(pert_generator.parameters(), lr=args.lr_Gp)

    lr_decay_steps = [0.6, 0.8]
    lr_decay_steps = [int(e * args.epochs) for e in lr_decay_steps]

    scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, lr_decay_steps, args.lr_decay)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, lr_decay_steps, args.lr_decay)
    scheduler_Gp = optim.lr_scheduler.MultiStepLR(optimizer_Gp, lr_decay_steps, args.lr_decay)


    for epoch in range(1, args.epochs + 1):
        # Train
        train(args, teacher=teacher, student=student, generator=generator,
              pert_generator=pert_generator,
              norm_trans=norm_trans, norm_trans_inv=norm_trans_inv,
              optimizer=[optimizer_S, optimizer_G,
                         optimizer_Gp
                         ], epoch=epoch, plotter=None,
              )

        scheduler_S.step()
        scheduler_G.step()
        scheduler_Gp.step()

        # # Test
        to_log_file("-" * 30 + "\n" + "Epoch {}:".format(epoch),args.checkpoint, 'contrast_log.txt')
        acc, asr = test_result(args, student, test_loader_clean, test_loader_dirty)


if __name__ == '__main__':
    main()


