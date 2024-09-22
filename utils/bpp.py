
from time import time

import numpy as np
import torch

import random
from numba import jit
from numba.types import float64, int64
import kornia.augmentation as A

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def back_to_np(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "gtsrb":
        expected_values = [0.3403, 0.3121, 0.3214]
        variance = [0.1595, 0.1590, 0.1683] 
    inputs_clone = inputs.clone()
    print(inputs_clone.shape)

    for channel in range(3):
        inputs_clone[channel, :, :] = inputs_clone[channel, :, :] * variance[channel] + expected_values[channel]
    return inputs_clone * 255


def back_to_np_4d(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "gtsrb":
        expected_values = [0.3403, 0.3121, 0.3214]
        variance = [0.1595, 0.1590, 0.1683]
    inputs_clone = inputs.clone()

    for channel in range(3):
        inputs_clone[:, channel, :, :] = inputs_clone[:, channel, :, :] * variance[channel] + expected_values[channel]

    return inputs_clone * 255

def np_4d_to_tensor(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "gtsrb":
        expected_values = [0.3403, 0.3121, 0.3214]
        variance = [0.1595, 0.1590, 0.1683]
    inputs_clone = inputs.clone().div(255.0)

    for channel in range(3):
        inputs_clone[:, channel, :, :] = (inputs_clone[:, channel, :, :] - expected_values[channel]).div(
            variance[channel])
    return inputs_clone


@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


@jit(nopython=True)
def floydDitherspeed(image, squeeze_num):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875
    return image


def train_bpp(opt, model, optimizer, train_dl, residual_list_train):

    print(" Train:")
    squeeze_num = opt.squeeze_num

    model.train()
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()

    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizer.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * opt.portion)
        num_neg = int(bs * opt.neg_rate)

        if num_bd != 0 and num_neg != 0:
            inputs_bd = back_to_np_4d(inputs[:num_bd], opt)
            if opt.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                        floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(opt.squeeze_num))).cuda())
            else:
                inputs_bd = torch.round(inputs_bd / 255.0 * (squeeze_num - 1)) / (squeeze_num - 1) * 255

            inputs_bd = np_4d_to_tensor(inputs_bd, opt)
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target

            inputs_negative = back_to_np_4d(inputs[num_bd: (num_bd + num_neg)], opt) + torch.cat(
                random.sample(residual_list_train, num_neg), dim=0)
            inputs_negative = torch.clamp(inputs_negative, 0, 255)
            inputs_negative = np_4d_to_tensor(inputs_negative, opt)

            total_inputs = torch.cat([inputs_bd, inputs_negative, inputs[(num_bd + num_neg):]], dim=0)
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

        elif (num_bd > 0 and num_neg == 0):
            inputs_bd = back_to_np_4d(inputs[:num_bd], opt)
            if opt.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                        floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(opt.squeeze_num))).cuda())
            else:
                inputs_bd = torch.round(inputs_bd / 255.0 * (squeeze_num - 1)) / (squeeze_num - 1) * 255

            inputs_bd = np_4d_to_tensor(inputs_bd, opt)
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target

            total_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

        elif (num_bd == 0 and num_neg == 0):
            total_inputs = inputs
            total_targets = targets

        total_inputs = transforms(total_inputs)
        start = time()
        total_preds,_ = model(total_inputs)
        total_time += time() - start
        loss_ce = criterion_CE(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizer.step()
        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_clean += bs - num_bd - num_neg
        total_bd += num_bd
        total_cross += num_neg
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_neg):], dim=1) == total_targets[(num_bd + num_neg):]
        )
        if num_bd:
            total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
            avg_acc_bd = total_bd_correct * 100.0 / total_bd
        else:
            avg_acc_bd = 0

        if num_neg:
            total_cross_correct += torch.sum(
                torch.argmax(total_preds[num_bd: (num_bd + num_neg)], dim=1)
                == total_targets[num_bd: (num_bd + num_neg)]
            )
            avg_acc_cross = total_cross_correct * 100.0 / total_cross
        else:
            avg_acc_cross = 0

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_loss_ce = total_loss_ce / total_sample
        
        if (num_bd > 0 and num_neg == 0):
 
            print("Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd)+ '\n')
        elif (num_bd > 0 and num_neg > 0):

            print("Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross)+ '\n')
        else:
            print("Loss: {:.4f} | Clean Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean)+ '\n')



def test_bpp(opt, model, test_dl,residual_list_test):
    squeeze_num = opt.squeeze_num
    model.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean,_ = model(inputs)

            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            inputs_bd = back_to_np_4d(inputs, opt)
            if opt.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                        floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(opt.squeeze_num))).cuda())
            else:
                inputs_bd = torch.round(inputs_bd / 255.0 * (squeeze_num - 1)) / (squeeze_num - 1) * 255

            inputs_bd = np_4d_to_tensor(inputs_bd, opt)
            targets_bd = torch.ones_like(targets) * opt.target
            preds_bd,_ = model(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.neg_rate:
                inputs_negative = back_to_np_4d(inputs, opt) + torch.cat(
                    random.sample(residual_list_test, inputs.shape[0]), dim=0)
                inputs_negative = np_4d_to_tensor(inputs_negative, opt)
                preds_cross,_ = model(inputs_negative)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample
    return acc_clean, acc_bd, acc_cross


def prepare_bpp(args,test_loader_clean):
    residual_list_test = []
    count = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader_clean):
        temp_negetive = back_to_np_4d(inputs, args)
        residual = torch.round(temp_negetive / 255.0 * (args.squeeze_num - 1)) / (
                args.squeeze_num - 1) * 255 - temp_negetive
        for i in range(residual.shape[0]):
            residual_list_test.append(residual[i].unsqueeze(0).cuda())
            count = count + 1

    return residual_list_test