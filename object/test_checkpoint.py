import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth, FocalLoss, FocalLossAdaptive, BrierScore, Entropy
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from torchmetrics.classification import MulticlassCalibrationError
from pprint import pprint

import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from reliability_diagrams import *

# Override matplotlib default styling.
plt.style.use("seaborn")

plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

plt.rc("axes", titlesize=16)
plt.rc("figure", titlesize=16)

def plot_reliability_diagram(
	y_true, 
	y_pred, 
	y_conf, 
	num_bins=10, 
	draw_ece=True, 
	draw_bin_importance="alpha", 
	draw_averages=True,
	title=None, 
	figsize=(6, 6), 
	dpi=100, 
	return_fig=True):
	fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
						  draw_bin_importance="alpha", draw_averages=True,
						  title=title, figsize=(6, 6), dpi=100, 
						  return_fig=True)
	# fig.savefig("temp_plot.png")
	return fig


def image_test(resize_size=256, crop_size=224, alexnet=False):
	if not alexnet:
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
	else:
		normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
	return  transforms.Compose([
		transforms.Resize((resize_size, resize_size)),
		transforms.CenterCrop(crop_size),
		transforms.ToTensor(),
		normalize
	])
 
def data_load(args): 
	## prepare data
	dsets = {}
	dset_loaders = {}
	train_bs = args.batch_size
	txt_test = open(args.test_dset_path).readlines()

	dsets["test"] = ImageList(txt_test, transform=image_test())
	dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*4, shuffle=True, num_workers=args.worker, drop_last=False)
	return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False, rel_title=None):
	start_test = True
	with torch.no_grad():
		iter_test = iter(loader)
		for i in tqdm(range(len(loader))):
			data = iter_test.next()
			inputs = data[0]
			labels = data[1]
			inputs = inputs.cuda()
			outputs = netC(netB(netF(inputs)))
			if start_test:
				all_output = outputs.float().cpu()
				all_label = labels.float()
				start_test = False
			else:
				all_output = torch.cat((all_output, outputs.float().cpu()), 0)
				all_label = torch.cat((all_label, labels.float()), 0)
	
	all_output = nn.Softmax(dim=1)(all_output)
	conf, predict = torch.max(all_output, 1)
	accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
	mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
 
	ent = mean_ent
	nll = nn.NLLLoss()(nn.LogSoftmax(dim=1)(all_output), all_label.long())
	bsc = BrierScore()(all_output, all_label.long())
	n_bins = 10
	ece = MulticlassCalibrationError(num_classes=args.class_num, n_bins=n_bins, norm="l1")(all_output, all_label)
	mce = MulticlassCalibrationError(num_classes=args.class_num, n_bins=n_bins, norm="max")(all_output, all_label)
	
	req_metrics = {
		"Entropy": ent,
		"NLL": nll.item(),
		"BSC": bsc.item(),
		"ECE": ece.item(),
		"MCE": mce.item(),
	}
 
	# pprint(
	# 	{
	# 		"Entropy": ent,
	# 		"NLL": nll.item(),
	# 		"BSC": bsc.item(),
	# 		"ECE": ece.item(),
	# 		"MCE": mce.item(),
	# 	}
	# )
	
	matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
	acc = matrix.diagonal()/matrix.sum(axis=1) * 100
	aacc = acc.mean()
	aa = [str(np.round(i, 2)) for i in acc]
	acc = ' '.join(aa)
	req_metrics["Acc"] = aacc
	
	rel_title += f" Acc: {aacc:.2f}" 
	rel_title += f" Ent: {ent:.3f}"
	rel_title += f" NLL: {nll.item():.3f}"
	rel_title += f" BSC: {bsc.item():.3f}"
	rel_title += f" MCE: {mce.item():.3f}"
	
 
	rel_fig = plot_reliability_diagram(
		all_label.cpu().detach().numpy(), 
		predict.cpu().detach().numpy(), 
		conf.cpu().detach().numpy(), 
		num_bins=10, 
		draw_ece=True, 
		draw_bin_importance="alpha", 
		draw_averages=True,
		title=rel_title, 
		figsize=(6, 6), 
		dpi=100, 
		return_fig=True)
 
	return aacc, acc, req_metrics, rel_fig
	
def load_model_state(model, ckpt_path):
	ckpt = torch.load(ckpt_path)
	list_a = list(model.state_dict().keys())
	list_b = list(ckpt.keys())
	for k in model.state_dict().keys():
		model.state_dict()[k] = ckpt["module." + k]
	return model

def compute_metrics(args):
	dset_loaders = data_load(args)
	## set base network
	if args.net[0:3] == 'res':
		netF = network.ResBase(res_name=args.net).cuda()
	elif args.net[0:3] == 'vgg':
		netF = network.VGGBase(vgg_name=args.net).cuda()  

	netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
	netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
	
	if args.src_only == 1:
		modelpath = args.ckpt_dir + '/source_F.pt'   
		netF.load_state_dict(torch.load(modelpath))
		modelpath = args.ckpt_dir + '/source_B.pt'   
		netB.load_state_dict(torch.load(modelpath))
		modelpath = args.ckpt_dir + '/source_C.pt'    
		netC.load_state_dict(torch.load(modelpath))
	else:
		files = [x for x in os.listdir(args.ckpt_dir) if x.startswith("target_")]
		modelpath = args.ckpt_dir + "/" + [x for x in files if "_F_" in x][0]
		netF.load_state_dict(torch.load(modelpath))
		modelpath = args.ckpt_dir + "/" + [x for x in files if "_B_" in x][0]
		netB.load_state_dict(torch.load(modelpath))
		modelpath = args.ckpt_dir + "/" + [x for x in files if "_C_" in x][0]
		netC.load_state_dict(torch.load(modelpath))
 
	netF.eval()
	netB.eval()
	netC.eval()
 
	acc_t_te, acc_t_list, req_metrics, rel_fig = cal_acc(dset_loaders['test'], netF, netB, netC, True, args.identifier)

	rel_fig.savefig(args.ckpt_dir + "/reliability_diagram.png")
	req_metrics["Per-Class Acc"] = acc_t_list
	with open(args.ckpt_dir + "/metrics.json", "w") as f:
		json.dump(req_metrics, f)

	print(args.identifier)
	pprint("Metrics are: ")
	pprint(req_metrics)
	pprint(acc_t_list)
	
	
def print_args(args):
	s = "==========================================\n"
	for arg, content in args.__dict__.items():
		s += "{}:{}\n".format(arg, content)
	return s

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='SHOT')
	parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
	parser.add_argument('--s', type=int, default=0, help="source")
	parser.add_argument('--t', type=int, default=1, help="target")
	parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
	parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
	parser.add_argument('--worker', type=int, default=2, help="number of workers")
	parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
	parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
	parser.add_argument('--net', type=str, default='resnet101', help="vgg16, resnet50, resnet101")
	parser.add_argument('--seed', type=int, default=2020, help="random seed")
	parser.add_argument('--bottleneck', type=int, default=256)
	parser.add_argument('--epsilon', type=float, default=1e-5)
	parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
	parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
	parser.add_argument('--smooth', type=float, default=0.1)   
	parser.add_argument('--output', type=str, default='san')
	parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
	parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
	parser.add_argument('--use_pasta', type=int, default=0, choices=[0, 1])
	parser.add_argument('--pasta_a', type=float, default=3.0)
	parser.add_argument('--pasta_b', type=float, default=0.25)
	parser.add_argument('--pasta_k', type=float, default=2.0)
	parser.add_argument('--gamma', type=float, default=2.0)
	parser.add_argument('--loss', type=str, default="label_smooth_ce", choices=["label_smooth_ce", "focal_loss", "adaptive_focal_loss"])
 
	parser.add_argument("--ckpt_dir", type=str, default="weight/target/")
	parser.add_argument("--src_only", type=int, default=1)
	parser.add_argument("--identifier", type=str, default="vanilla_source")
	args = parser.parse_args()
 
	if args.dset == 'office-home':
		names = ['Art', 'Clipart', 'Product', 'RealWorld']
		args.class_num = 65 
	if args.dset == 'office':
		names = ['amazon', 'dslr', 'webcam']
		args.class_num = 31
	if args.dset == 'VISDA-C':
		names = ['train', 'validation']
		args.class_num = 12
	if args.dset == 'office-caltech':
		names = ['amazon', 'caltech', 'dslr', 'webcam']
		args.class_num = 10

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
	SEED = args.seed
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	torch.backends.cudnn.deterministic = True
 
	if args.dset == 'VISDA-C':
		args.s_dset_path = 'image_lists/visda/train_imagelist.txt'
		args.test_dset_path = 'image_lists/visda/validation_imagelist.txt'
	else: 
		folder = './data/'
		args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
		args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
		
	compute_metrics(args)   
   