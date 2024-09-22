from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
from typing import Union
from .utils import EnsembleNet, sigmoid
import random
from .utils import to_log_file

def shuffle_ckpt_layer(model, n_layers, is_alexnet=False, inplace=False):
    if not inplace:
        model = deepcopy(model)
    model_state = model.state_dict()

    total_num_layer = 0
    for k, v in model_state.items():
        if 'conv' in k or (is_alexnet and len(v.shape) == 4):
            total_num_layer += 1
    if n_layers > 0:
        shuffle_index = [0]*(total_num_layer-n_layers) + [1] * n_layers
    elif n_layers < 0:
        shuffle_index = [1] * abs(n_layers) + [0] * \
            (total_num_layer-abs(n_layers))
    else:
        return model
    new_ckpt = {}
    i = 0
    for k, v in model_state.items():
        if ('conv' in k and 'bias' not in k) or (is_alexnet and len(v.shape) == 4):
            if shuffle_index[i] == 1:
                _, channels, _, _ = v.size()
                idx = torch.randperm(channels)
                v = v[:, idx, ...]
            i += 1
        new_ckpt[k] = v
    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model


class BackdoorSuspectLoss(nn.Module):
    def __init__(self, model, coef=1., n_shuf_ens=3, n_shuf_layer=4, no_shuf_grad=False,
                 device='cuda', batch_size=128, args=None):
        super().__init__()
        self.model = model
        self.args = args
        self.coef = coef
        self.n_shuf_ens = n_shuf_ens
        self.n_shuf_layer = n_shuf_layer
        self.no_shuf_grad = no_shuf_grad
        self.device = device
        self.shufl_model = self.make_shuffle_suspect(model)
        self.n_shufl_penalized = 0
        self.pseudo_test_flag = False

    def make_shuffle_suspect(self, model):
        shuf_models = []
        # NOTE ensemble shuffled models to get a stable prediction. Otherwise, the prediction highly depends on seeds.
        for _ in range(self.n_shuf_ens):
            t = shuffle_ckpt_layer(
                model, self.n_shuf_layer)
            shuf_models.append(t)
        shuffle_model = EnsembleNet(shuf_models)
        return shuffle_model

    def test_suspect_model(self, test_loader, test_poi_loader):
        poi_kl_loss, cl_kl_loss = self.compare_shuffle(
            self.model, self.shufl_model, test_loader, test_poi_loader)
        return poi_kl_loss, cl_kl_loss


    def loss(self, logits, x_pseudo) -> Union[float, torch.Tensor]:
        if self.coef == 0.:
            return 0.

        self.shufl_model.eval()
        if self.no_shuf_grad:
            with torch.no_grad():
                shufl_model_logits,*_ = self.shufl_model(x_pseudo)
        else:
            shufl_model_logits,*_ = self.shufl_model(x_pseudo)
            
        shufl_mask = shufl_model_logits.max(1)[1].eq(logits.max(1)[1]).float()
        
        if shufl_mask.sum() > 0.:
            shufl_kl_loss = self.divergence(
                shufl_model_logits, logits, reduction='none')
            shufl_kl_loss = shufl_kl_loss.mean(1) 
            shufl_kl_loss = torch.sum(shufl_kl_loss * shufl_mask) / shufl_mask.sum()
            self.n_shufl_penalized = shufl_mask.sum().item()
        else:
            shufl_kl_loss = 0.
            self.n_shufl_penalized = 0
        return shufl_mask, - self.coef * shufl_kl_loss


    def compute_divergences(self, model, shuffle_model, loader) -> list:
        kl_loss = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, *_ = model(x)
                sh_logits, *_ = shuffle_model(x)
                divergence_loss = self.divergence(
                    sh_logits, logits, reduction='none')
                divergence_loss = divergence_loss.sum(1).data.cpu().numpy()
                kl_loss.extend(divergence_loss.tolist())
        return kl_loss

    def compare_shuffle(self, model, shuffle_model, test_loader, test_poi_loader):
        model.eval()
        poi_kl_loss = self.compute_divergences(
            model, shuffle_model, test_poi_loader)
        to_log_file(
            f" Poison KL Loss: {np.mean(poi_kl_loss):.3f}",
            self.args.checkpoint, 'defense_log.txt')

        cln_kl_loss = self.compute_divergences(
            model, shuffle_model, test_loader)
        to_log_file(
            f" Clean KL Loss: {np.mean(cln_kl_loss):.3f}",
            self.args.checkpoint, 'defense_log.txt')

        all_loss = np.array(poi_kl_loss + cln_kl_loss)
        all_scores = sigmoid(all_loss - np.mean(all_loss))
        all_labels = [1] * len(poi_kl_loss) + [0] * len(cln_kl_loss)

        acc = accuracy_score(all_labels, all_scores < 0.5)
        auc = roc_auc_score(all_labels, all_scores)
        to_log_file(
            f" Poi acc: {acc*100:.1f}% | auc: {auc*100:.1f}% | std: {np.std(all_loss):.3f}",
            self.args.checkpoint, 'defense_log.txt')

        return np.mean(poi_kl_loss), np.mean(cln_kl_loss)

    def divergence(self, student_logits, teacher_logits, reduction='mean'):
        divergence = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(
            teacher_logits, dim=1), reduction=reduction)
        return divergence
