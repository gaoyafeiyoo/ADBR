import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import argparse
import os
import sys
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.bpp import *
from utils import *
from utils.get_model_loader import *
from utils.utils import *
from utils.wanet import *
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from itertools import repeat
from typing import List, Callable
from torch import Tensor
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader, random_split, RandomSampler

'''
Based on the paper 'On the Iteration Complexity of Hypergradient Computation,' this code was created.
Source: https://github.com/prolearner/hypertorch/blob/master/hypergrad/hypergradients.py
Original Author: Riccardo Grazzi
'''

def get_args():
	# set the basic parameter
	parser = argparse.ArgumentParser()
	# basic
	parser.add_argument("--seed", type=int, default=0) 
	parser.add_argument("--mode", type=str, default='contrast')
	parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
	parser.add_argument('--attack', type=str, default='patch', choices=['patch', 'wanet', 'bpp', 'blended', 'sig'])
	# parser.add_argument('--pattern', type=str, default='grid', choices=['white', 'grid', 'color'])
	parser.add_argument('--target', type=int, default=0) 
	parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'pt_resnet'])
	parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
	parser.add_argument('--channels', type=int, default=3, help='number of image channels')
	parser.add_argument('--portion', type=float, default=0.1)

	parser.add_argument('--lr', type=float,default=0.0001)
	parser.add_argument('--data', type=str, default='/home/jovyan/exp_3145/cache/data')
	parser.add_argument('--output_dir', type=str, default='./cache/weights/')
	parser.add_argument('--checkpoint_root', type=str, default='./checkpoint_root')
	parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
	parser.add_argument('--method', type=str, default='i-BAU', choices=['CLP', 'NAD', 'ANP','i-BAU','MCR'])

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
	# parser.add_argument('--pt', type=bool, default=True)
    # for bpp
	parser.add_argument("--neg_rate", type=float, default=0.2)
	parser.add_argument("--squeeze_num", type=int, default=8)
	parser.add_argument("--dithering", type=bool, default=False)

	# set the parameter for the i-bau defense
	parser.add_argument('--ratio', type=float,default=0.05, help='the ratio of clean data loader')
	## hyper params
	### TODO config optimizer 改框架之后放到前面统一起来
	parser.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')
	parser.add_argument('--n_rounds', type=int,default=5, help='the maximum number of unelarning rounds')
	parser.add_argument('--K', type=int,default=5,help='the maximum number of fixed point iterations')

	args = parser.parse_args()
	return args



class DifferentiableOptimizer:
	def __init__(self, loss_f, dim_mult, data_or_iter=None):
		"""
		Args:
			loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
			data_or_iter: (x, y) or iterator over the data needed for loss_f
		"""
		self.data_iterator = None
		if data_or_iter:
			self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(data_or_iter)

		self.loss_f = loss_f
		self.dim_mult = dim_mult
		self.curr_loss = None

	def get_opt_params(self, params):
		opt_params = [p for p in params]
		opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult-1) ])
		return opt_params

	def step(self, params, hparams, create_graph):
		raise NotImplementedError

	def __call__(self, params, hparams, create_graph=True):
		with torch.enable_grad():
			return self.step(params, hparams, create_graph)

	def get_loss(self, params, hparams):
		if self.data_iterator:
			data = next(self.data_iterator)
			self.curr_loss = self.loss_f(params, hparams, data)
		else:
			self.curr_loss = self.loss_f(params, hparams)
		return self.curr_loss


class GradientDescent(DifferentiableOptimizer):
	def __init__(self, loss_f, step_size, data_or_iter=None):
		super(GradientDescent, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
		self.step_size_f = step_size if callable(step_size) else lambda x: step_size

	def step(self, params, hparams, create_graph):
		loss = self.get_loss(params, hparams)
		sz = self.step_size_f(hparams)
		return gd_step(params, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
	grads = torch.autograd.grad(loss, params, create_graph=create_graph)
	return [w - step_size * g for w, g in zip(params, grads)]


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
	grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
								retain_graph=retain_graph, create_graph=create_graph)

	def grad_or_zeros(grad, var):
		return torch.zeros_like(var) if grad is None else grad

	return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
	grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
	grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

	return grad_outer_w, grad_outer_hparams

def update_tensor_grads(hparams, grads):
	for l, g in zip(hparams, grads):
		if l.grad is None:
			l.grad = torch.zeros_like(l)
		if g is not None:
			l.grad += g

def fixed_point(params: List[Tensor],
				hparams: List[Tensor],
				K: int ,
				fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
				outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
				tol=1e-10,
				set_grad=True,
				stochastic=False) -> List[Tensor]:
	"""
	Computes the hypergradient by applying K steps of the fixed point method (it can end earlier when tol is reached).
	Args:
		params: the output of the inner solver procedure.
		hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
		K: the maximum number of fixed point iterations
		fp_map: the fixed point map which defines the inner problem
		outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
		tol: end the method earlier when  the normed difference between two iterates is less than tol
		set_grad: if True set t.grad to the hypergradient for every t in hparams
		stochastic: set this to True when fp_map is not a deterministic function of its inputs
	Returns:
		the list of hypergradients for each element in hparams
	"""

	params = [w.detach().requires_grad_(True) for w in params]
	o_loss = outer_loss(params, hparams)
	grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

	if not stochastic:
		w_mapped = fp_map(params, hparams)

	vs = [torch.zeros_like(w) for w in params]
	vs_vec = cat_list_to_tensor(vs)
	for k in range(K):
		vs_prev_vec = vs_vec

		if stochastic:
			w_mapped = fp_map(params, hparams)
			vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
		else:
			vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

		vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
		vs_vec = cat_list_to_tensor(vs)
		if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
			break

	if stochastic:
		w_mapped = fp_map(params, hparams)

	grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
	grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

	if set_grad:
		update_tensor_grads(hparams, grads)
	return grads


def cat_list_to_tensor(list_tx):
	return torch.cat([xx.reshape([-1]) for xx in list_tx])

def i_bau(args):

	### define the inner loss L2
	def loss_inner(perturb, model_params):
		### TODO: cpu training and multiprocessing
		images = images_list[0].to(args.device)
		labels = labels_list[0].long().to(args.device)
		#per_img = torch.clamp(images+perturb[0],min=0,max=1)
		per_img = images+perturb[0]
		per_logits, _ = model.forward(per_img)
		loss = F.cross_entropy(per_logits, labels, reduction='none')
		loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
		return loss_regu

	### define the outer loss L1
	def loss_outer(perturb, model_params):
		### TODO: cpu training and multiprocessing
		portion = 0.01
		images, labels = images_list[batchnum].to(args.device), labels_list[batchnum].long().to(args.device)
		patching = torch.zeros_like(images, device='cuda')
		number = images.shape[0]
		rand_idx = random.sample(list(np.arange(number)), int(number*portion))
		patching[rand_idx] = perturb[0]
		#unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
		unlearn_imgs = images + patching
		logits,_ = model(unlearn_imgs)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(logits, labels)
		return loss

	# 1.Prepare and test poison model, optimizer, scheduler
	model, data_test_loader_clean, data_test_loader_dirty = get_defense_loader(args)
	model.to(args.device)

	### TODO: adam and sgd
	outer_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

	# 2. get some clean data
	print("We use clean train data, the original paper use clean test data.")
	transforms_list = []
	transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
	transforms_list.append(transforms.ToTensor())
	transforms_list.append(get_Normalize(args))
	transform_train = transforms.Compose(transforms_list)
	if args.dataset == 'cifar10':
		data_train = CIFAR10(args.data+'/cifar10/',train=True,download=False,transform=transform_train)
	elif args.dataset == 'gtsrb':
		data_train = GTSRB(args,train=True,transforms=transform_train)
	random_sampler = RandomSampler(data_source=data_train, replacement=False,num_samples=int(args.ratio * len(data_train)))
	train_loader = DataLoader(data_train, batch_size=args.batch_size,sampler=random_sampler, num_workers=8)
	images_list, labels_list = [], []
	for index, (images, labels) in enumerate(train_loader):
		images_list.append(images)
		labels_list.append(labels)
	inner_opt = GradientDescent(loss_inner, 0.1)


	# b. unlearn the backdoor model by the pertubation
	print("=> Conducting Defence..")
	model.eval()
	best_ACC = 0
	best_ASR = 0
	for round in range(args.n_rounds):
		# batch_pert = torch.zeros_like(data_clean_testset[0][:1], requires_grad=True, device=args.device)
		batch_pert = torch.zeros([1, args.input_channel, args.input_height, args.input_width], requires_grad=True, device=args.device)
		batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)

		for images, labels in train_loader:
			images = images.to(args.device)
			
			ori_lab = torch.argmax(model.forward(images,out_feature=False), axis=1).long()
			# per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
			per_logits = model.forward(images+batch_pert,out_feature=False)
			loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
			loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(batch_pert), 2)
			batch_opt.zero_grad()
			loss_regu.backward(retain_graph=True)
			batch_opt.step()

		#l2-ball
		# pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
		pert = batch_pert

		#unlearn step         
		for batchnum in range(len(images_list)): 
			outer_opt.zero_grad()
			fixed_point(pert, list(model.parameters()), args.K, inner_opt, loss_outer) 
			outer_opt.step()

		ACC, ASR = test_result(args,model,data_test_loader_clean,data_test_loader_dirty,round)
		
		if best_ACC < ACC:
			best_ACC = ACC
			best_ASR = ASR

	to_log_file('Best Test Acc: {:.3f}%'.format(best_ACC),
				args.checkpoint, 'contrast_log.txt')
	to_log_file('Best Test Asr: {:.3f}%'.format(best_ASR) + "\n",
				args.checkpoint, 'contrast_log.txt')

	result = {}
	result['model'] = model
	return result


def test_result(args,net,data_test_loader_clean,data_test_loader_dirty,epoch):
	if args.attack == 'wanet':
		path = os.path.join(args.output_dir,
								f'{args.model}' + '_' + f'{args.attack}' + '_' + f'{args.dataset}.pth')
		state = torch.load(path)
		identity_grid = state['identity_grid']
		noise_grid = state['noise_grid']
		acc, asr, acc_cross, _ = test_wanet(args, net, data_test_loader_clean, noise_grid, identity_grid)
		to_log_file(
			'Epoch: %d Clean acc: %.2f BD asr: %.2f Cross acc: %.2f ' % (epoch, acc, asr, acc_cross) + "\n",
			args.checkpoint, 'contrast_log.txt')

	elif args.attack == 'bpp':
		residual_list_test = prepare_bpp(args,data_test_loader_clean)
		acc, asr, acc_cross = test_bpp(args, net, data_test_loader_clean, residual_list_test)
		to_log_file(
			'Epoch: %d Clean acc: %.2f BD asr: %.2f Cross acc: %.2f ' % (epoch, acc, asr, acc_cross),
			args.checkpoint, 'defense_log.txt')

	else:
		clean_loss, acc = test(net, data_test_loader_clean)
		dirty_loss, asr = test(net, data_test_loader_dirty)
		to_log_file(
			'Epoch: %d model clean_loss: %.4f, acc: %.2f' % (epoch, clean_loss, acc),
			args.checkpoint, 'contrast_log.txt')
		to_log_file(
			'Epoch: %d model dirty_loss: %.4f, asr: %.2f' % (epoch, dirty_loss, asr) + "\n",
			args.checkpoint, 'contrast_log.txt')
	return acc,asr



if __name__ == '__main__':

	### 1. basic setting: args
	args = get_args()
	set_random_seed(args)
	more_config(args)

	###2.load and test poison model
	model, data_test_loader_clean, data_test_loader_dirty = get_defense_loader(args)
	test_result(args,model,data_test_loader_clean,data_test_loader_dirty,0)

	### 3. i-bau defense:
	result_defense = i_bau(args)

	### 4. test the defense result
	bau_model = result_defense['model'].eval()
	bau_model.to(args.device)
	test_result(args,bau_model,data_test_loader_clean,data_test_loader_dirty,0)