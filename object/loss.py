import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

from scipy.special import lambertw

def Entropy(input_):
	bs = input_.size(0)
	epsilon = 1e-5
	entropy = -input_ * torch.log(input_ + epsilon)
	entropy = torch.sum(entropy, dim=1)
	return entropy 

def grl_hook(coeff):
	def fun1(grad):
		return -coeff*grad.clone()
	return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
	softmax_output = input_list[1].detach()
	feature = input_list[0]
	if random_layer is None:
		op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
		ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
	else:
		random_out = random_layer.forward([feature, softmax_output])
		ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
	batch_size = softmax_output.size(0) // 2
	dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
	if entropy is not None:
		entropy.register_hook(grl_hook(coeff))
		entropy = 1.0+torch.exp(-entropy)
		source_mask = torch.ones_like(entropy)
		source_mask[feature.size(0)//2:] = 0
		source_weight = entropy*source_mask
		target_mask = torch.ones_like(entropy)
		target_mask[0:feature.size(0)//2] = 0
		target_weight = entropy*target_mask
		weight = source_weight / torch.sum(source_weight).detach().item() + \
				 target_weight / torch.sum(target_weight).detach().item()
		return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
	else:
		return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net):
	ad_out = ad_net(features)
	batch_size = ad_out.size(0) // 2
	dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
	return nn.BCELoss()(ad_out, dc_target)


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.
	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.
	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.reduction = reduction
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
		if self.use_gpu: targets = targets.cuda()
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).sum(dim=1)
		if self.reduction:
			return loss.mean()
		else:
			return loss
		return loss
	
class FocalLoss(nn.Module):
	def __init__(self, gamma=0, alpha=None, use_gpu=True, reduction=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		# if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
		# if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
		self.use_gpu = use_gpu
		self.reduction = reduction

	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)
		if self.use_gpu: target = target.cuda()

		logpt = F.log_softmax(input)
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type()!=input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0,target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1-pt)**self.gamma * logpt
		if self.reduction: return loss.mean()
		else: return loss.sum()

def get_gamma(p=0.2):
	'''
	Get the gamma for a given pt where the function g(p, gamma) = 1
	'''
	y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
	gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
	gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
	return gamma

ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
	gamma_dic[p] = gammas[i]
	i += 1

class FocalLossAdaptive(nn.Module):
	def __init__(self, gamma=0, use_gpu=True, reduction=True):
		super(FocalLossAdaptive, self).__init__()
		self.gamma = gamma
		self.use_gpu = use_gpu
		self.reduction = reduction

	def get_gamma_list(self, pt):
		gamma_list = []
		batch_size = pt.shape[0]
		for i in range(batch_size):
			pt_sample = pt[i].item()
			if (pt_sample >= 0.5):
				gamma_list.append(self.gamma)
				continue
			# Choosing the gamma for the sample
			for key in sorted(gamma_dic.keys()):
				if pt_sample < key:
					gamma_list.append(gamma_dic[key])
					break
		return torch.tensor(gamma_list).cuda()

	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)
		if self.use_gpu: target = target.cuda()
		logpt = F.log_softmax(input, dim=1)
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = logpt.exp()
		gamma = self.get_gamma_list(pt)
		loss = -1 * (1-pt)**gamma * logpt
		if self.reduction: return loss.mean()
		else: return loss.sum()

class BrierScore(nn.Module):
	def __init__(self, use_gpu=True):
		super(BrierScore, self).__init__()
		self.use_gpu = use_gpu

	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)
		if self.use_gpu: target = target.cuda()
		target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
		target_one_hot.zero_()
		target_one_hot.scatter_(1, target, 1)

		pt = F.softmax(input, dim=-1)
		squared_diff = (target_one_hot - pt.to(target.get_device())) ** 2

		loss = torch.sum(squared_diff) / float(input.shape[0])
		return loss