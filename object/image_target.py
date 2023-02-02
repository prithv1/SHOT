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
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import CrossEntropyLabelSmooth, FocalLoss, FocalLossAdaptive, BrierScore, Entropy
from torchmetrics.classification import MulticlassCalibrationError
from pprint import pprint

class PASTA:
	def __init__(self, alpha: float = 3, beta: float = 0.25, k: int = 2):
		self.alpha = alpha
		self.beta = beta
		self.k = k
	
	def __call__(self, img):
		img = transforms.ToTensor()(img)
		fft_src = torch.fft.fftn(img, dim=[-2, -1])
		amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

		X, Y = amp_src.shape[1:]
		X_range, Y_range = None, None

		if X % 2 == 1:
			X_range = np.arange(-1 * (X // 2), (X // 2) + 1)
		else:
			X_range = np.concatenate(
				[np.arange(-1 * (X // 2) + 1, 1), np.arange(0, X // 2)]
			)

		if Y % 2 == 1:
			Y_range = np.arange(-1 * (Y // 2), (Y // 2) + 1)
		else:
			Y_range = np.concatenate(
				[np.arange(-1 * (Y // 2) + 1, 1), np.arange(0, Y // 2)]
			)

		XX, YY = np.meshgrid(Y_range, X_range)

		exp = self.k
		lin = self.alpha
		offset = self.beta

		inv = np.sqrt(np.square(XX) + np.square(YY))
		inv *= (1 / inv.max()) * lin
		inv = np.power(inv, exp)
		inv = np.tile(inv, (3, 1, 1))
		inv += offset
		prop = np.fft.fftshift(inv, axes=[-2, -1])
		amp_src = amp_src * np.random.normal(np.ones(prop.shape), prop)

		aug_img = amp_src * torch.exp(1j * pha_src)
		aug_img = torch.fft.ifftn(aug_img, dim=[-2, -1])
		aug_img = torch.real(aug_img)
		aug_img = torch.clip(aug_img, 0, 1)
		aug_img = transforms.ToPILImage()(aug_img)
		return aug_img

def op_copy(optimizer):
	for param_group in optimizer.param_groups:
		param_group['lr0'] = param_group['lr']
	return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
	decay = (1 + gamma * iter_num / max_iter) ** (-power)
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr0'] * decay
		param_group['weight_decay'] = 1e-3
		param_group['momentum'] = 0.9
		param_group['nesterov'] = True
	return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False, use_pasta=0, pasta_args=None):
	if not alexnet:
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
	else:
		normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
	if use_pasta == 1:
		print("Using PASTA Aug")
		return  transforms.Compose([
			PASTA(alpha=pasta_args["a"], beta=pasta_args["b"], k=pasta_args["k"]),
			transforms.Resize((resize_size, resize_size)),
			transforms.RandomCrop(crop_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
	else:
		return  transforms.Compose([
			transforms.Resize((resize_size, resize_size)),
			transforms.RandomCrop(crop_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])

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
	txt_tar = open(args.t_dset_path).readlines()
	txt_test = open(args.test_dset_path).readlines()

	if not args.da == 'uda':
		label_map_s = {}
		for i in range(len(args.src_classes)):
			label_map_s[args.src_classes[i]] = i

		new_tar = []
		for i in range(len(txt_tar)):
			rec = txt_tar[i]
			reci = rec.strip().split(' ')
			if int(reci[1]) in args.tar_classes:
				if int(reci[1]) in args.src_classes:
					line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
					new_tar.append(line)
				else:
					line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
					new_tar.append(line)
		txt_tar = new_tar.copy()
		txt_test = txt_tar.copy()

	dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
	dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
	dsets["test"] = ImageList_idx(txt_test, transform=image_test())
	dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

	return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
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
	_, predict = torch.max(all_output, 1)
	accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
	mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

	ent = mean_ent
	nll = nn.NLLLoss()(nn.LogSoftmax(dim=1)(all_output), all_label.long())
	bsc = BrierScore()(all_output, all_label.long())*100.0
	n_bins = 10
	ece = MulticlassCalibrationError(num_classes=args.class_num, n_bins=n_bins, norm="l1")(all_output, all_label)*100.0
	mce = MulticlassCalibrationError(num_classes=args.class_num, n_bins=n_bins, norm="max")(all_output, all_label)*100.0

	pprint(
		{
			"Entropy": ent,
			"NLL": nll.item(),
			"BSC": bsc.item(),
			"ECE": ece.item(),
			"MCE": mce.item(),
		}
	)
	
	req_metrics = {
		"Entropy": ent,
		"NLL": nll.item(),
		"BSC": bsc.item(),
		"ECE": ece.item(),
		"MCE": mce.item(),
	}

	if flag:
		matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
		acc = matrix.diagonal()/matrix.sum(axis=1) * 100
		aacc = acc.mean()
		aa = [str(np.round(i, 2)) for i in acc]
		acc = ' '.join(aa)
		return aacc, acc, req_metrics
	else:
		return accuracy*100, mean_ent

def train_target(args):
	dset_loaders = data_load(args)
	## set base network
	if args.net[0:3] == 'res':
		netF = network.ResBase(res_name=args.net).cuda()
	elif args.net[0:3] == 'vgg':
		netF = network.VGGBase(vgg_name=args.net).cuda()  

	netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
	netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

	modelpath = args.output_dir_src + '/source_F.pt'   
	netF.load_state_dict(torch.load(modelpath))
	modelpath = args.output_dir_src + '/source_B.pt'   
	netB.load_state_dict(torch.load(modelpath))
	modelpath = args.output_dir_src + '/source_C.pt'    
	netC.load_state_dict(torch.load(modelpath))
	netC.eval()
	for k, v in netC.named_parameters():
		v.requires_grad = False

	param_group = []
	for k, v in netF.named_parameters():
		if args.lr_decay1 > 0:
			param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
		else:
			v.requires_grad = False
	for k, v in netB.named_parameters():
		if args.lr_decay2 > 0:
			param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
		else:
			v.requires_grad = False

	optimizer = optim.SGD(param_group)
	optimizer = op_copy(optimizer)

	max_iter = args.max_epoch * len(dset_loaders["target"])
	interval_iter = max_iter // args.interval
	iter_num = 0

	while iter_num < max_iter:
		try:
			inputs_test, labels_test, tar_idx = iter_test.next()
		except:
			iter_test = iter(dset_loaders["target"])
			inputs_test, labels_test, tar_idx = iter_test.next()

		if inputs_test.size(0) == 1:
			continue

		if iter_num % interval_iter == 0 and args.cls_par > 0:
			netF.eval()
			netB.eval()
			mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
			mem_label = torch.from_numpy(mem_label).cuda()
			netF.train()
			netB.train()

		inputs_test = inputs_test.cuda()

		iter_num += 1
		lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

		features_test = netB(netF(inputs_test))
		outputs_test = netC(features_test)

		if args.cls_par > 0:
			pred = mem_label[tar_idx]
			if args.loss == "ce":
				loss_str = "CE"
				classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
				classifier_loss *= args.cls_par
			elif args.loss == "focal_loss":
				loss_str = "Focal"
				classifier_loss = FocalLoss(gamma=args.gamma)(outputs_test, pred)
				classifier_loss *= args.cls_par
			elif args.loss == "adaptive_focal_loss":
				loss_str = "AdaFocal"
				classifier_loss = FocalLossAdaptive(gamma=args.gamma)(outputs_test, pred)
				classifier_loss *= args.cls_par
				
			if iter_num < interval_iter and args.dset == "VISDA-C":
				classifier_loss *= 0
		else:
			classifier_loss = torch.tensor(0.0).cuda()

		if args.ent:
			softmax_out = nn.Softmax(dim=1)(outputs_test)
			entropy_loss = torch.mean(loss.Entropy(softmax_out))
			if args.gent:
				msoftmax = softmax_out.mean(dim=0)
				gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
				entropy_loss -= gentropy_loss
			im_loss = entropy_loss * args.ent_par
			classifier_loss += im_loss
		
		# outputs_test = outputs_test.to(labels_test.get_device())
		ent = Entropy(nn.Softmax(dim=1)(outputs_test)).mean()
		nll = nn.NLLLoss()(nn.LogSoftmax(dim=1)(outputs_test), labels_test.cuda())
		bsc = BrierScore()(outputs_test, labels_test)*100.0
		n_bins = 10
		ece = MulticlassCalibrationError(num_classes=args.class_num, n_bins=n_bins, norm="l1").cuda()(outputs_test, labels_test.cuda())*100.0
		mce = MulticlassCalibrationError(num_classes=args.class_num, n_bins=n_bins, norm="max").cuda()(outputs_test, labels_test.cuda())*100.0
   
		if iter_num % 50 == 0:
			log_str = 'Adaptation, Iter:{}; {} Loss = {:.2f} NLL = {:.4f} Ent = {:.4f} BSC = {:.4f} ECE = {:.4f} MCE = {:.4f}'.format(iter_num, loss_str, classifier_loss.item(), nll.item(), ent.item(), bsc.item(), ece, mce)
			args.out_file.write(log_str + '\n')
			args.out_file.flush()
			print(log_str)

		optimizer.zero_grad()
		classifier_loss.backward()
		optimizer.step()

		if iter_num % interval_iter == 0 or iter_num == max_iter:
			netF.eval()
			netB.eval()
			if args.dset=='VISDA-C':
				acc_s_te, acc_list, req_metrics = cal_acc(dset_loaders['test'], netF, netB, netC, True)
				log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
				log_str += "\n Target: " + str(req_metrics)
			else:
				acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
				log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

			args.out_file.write(log_str + '\n')
			args.out_file.flush()
			print(log_str+'\n')
			
			torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_ep_" + str(iter_num) + ".pt"))
			torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_ep_" + str(iter_num) + ".pt"))
			torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_ep_" + str(iter_num) + ".pt"))
			
			netF.train()
			netB.train()

	if args.issave:   
		torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
		torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
		torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
		
	return netF, netB, netC

def print_args(args):
	s = "==========================================\n"
	for arg, content in args.__dict__.items():
		s += "{}:{}\n".format(arg, content)
	return s

def test_target(args):
	dset_loaders = data_load(args)
	## set base network
	if args.net[0:3] == 'res':
		netF = network.ResBase(res_name=args.net).cuda()
	elif args.net[0:3] == 'vgg':
		netF = network.VGGBase(vgg_name=args.net).cuda()  

	netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
	netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
	
	args.modelpath = args.output_dir_src + '/target_F_' + args.savename +'.pt'   
	netF.load_state_dict(torch.load(args.modelpath))
	args.modelpath = args.output_dir_src + '/target_B_' + args.savename +'.pt'   
	netB.load_state_dict(torch.load(args.modelpath))
	args.modelpath = args.output_dir_src + '/target_C_' + args.savename +'.pt'   
	netC.load_state_dict(torch.load(args.modelpath))
	netF.eval()
	netB.eval()
	netC.eval()

	if args.da == 'oda':
		acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
		log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
	else:
		if args.dset=='VISDA-C':
			acc, acc_list, req_metrics = cal_acc(dset_loaders['test'], netF, netB, netC, True)
			log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
			log_str += "\n Target: " + str(req_metrics)
		else:
			acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
			log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

	args.out_file.write(log_str)
	args.out_file.flush()
	print(log_str)

def obtain_label(loader, netF, netB, netC, args):
	start_test = True
	with torch.no_grad():
		iter_test = iter(loader)
		print("Obtaining Labels..")
		for _ in tqdm(range(len(loader))):
			data = iter_test.next()
			inputs = data[0]
			labels = data[1]
			inputs = inputs.cuda()
			feas = netB(netF(inputs))
			outputs = netC(feas)
			if start_test:
				all_fea = feas.float().cpu()
				all_output = outputs.float().cpu()
				all_label = labels.float()
				start_test = False
			else:
				all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
				all_output = torch.cat((all_output, outputs.float().cpu()), 0)
				all_label = torch.cat((all_label, labels.float()), 0)

	all_output = nn.Softmax(dim=1)(all_output)
	ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
	unknown_weight = 1 - ent / np.log(args.class_num)
	_, predict = torch.max(all_output, 1)

	accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
	if args.distance == 'cosine':
		all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
		all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

	all_fea = all_fea.float().cpu().numpy()
	K = all_output.size(1)
	aff = all_output.float().cpu().numpy()

	for _ in range(2):
		initc = aff.transpose().dot(all_fea)
		initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
		cls_count = np.eye(K)[predict].sum(axis=0)
		labelset = np.where(cls_count>args.threshold)
		labelset = labelset[0]

		dd = cdist(all_fea, initc[labelset], args.distance)
		pred_label = dd.argmin(axis=1)
		predict = labelset[pred_label]

		aff = np.eye(K)[predict]

	acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
	log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

	args.out_file.write(log_str + '\n')
	args.out_file.flush()
	print(log_str+'\n')

	return predict.astype('int')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='SHOT')
	parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
	parser.add_argument('--s', type=int, default=0, help="source")
	parser.add_argument('--t', type=int, default=1, help="target")
	parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
	parser.add_argument('--interval', type=int, default=15)
	parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
	parser.add_argument('--worker', type=int, default=4, help="number of workers")
	parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
	parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
	parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
	parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
	parser.add_argument('--gent', type=bool, default=True)
	parser.add_argument('--ent', type=bool, default=True)
	parser.add_argument('--threshold', type=int, default=0)
	parser.add_argument('--cls_par', type=float, default=0.3)
	parser.add_argument('--ent_par', type=float, default=1.0)
	parser.add_argument('--lr_decay1', type=float, default=0.1)
	parser.add_argument('--lr_decay2', type=float, default=1.0)

	parser.add_argument('--bottleneck', type=int, default=256)
	parser.add_argument('--epsilon', type=float, default=1e-5)
	parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
	parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
	parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
	parser.add_argument('--output', type=str, default='san')
	parser.add_argument('--output_src', type=str, default='san')
	parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
	parser.add_argument('--issave', type=bool, default=True)
	
	parser.add_argument('--use_pasta', type=int, default=0, choices=[0, 1])
	parser.add_argument('--pasta_a', type=float, default=3.0)
	parser.add_argument('--pasta_b', type=float, default=0.25)
	parser.add_argument('--pasta_k', type=float, default=2.0)
	parser.add_argument('--gamma', type=float, default=2.0)
	parser.add_argument('--loss', type=str, default="ce", choices=["ce", "focal_loss", "adaptive_focal_loss"])
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
	# torch.backends.cudnn.deterministic = True

	for i in range(len(names)):
		if i == args.s:
			continue
		args.t = i

		if args.dset == 'VISDA-C':
			args.s_dset_path = 'image_lists/visda/train_imagelist.txt'
			args.t_dset_path = 'image_lists/visda/validation_imagelist.txt'
			args.test_dset_path = 'image_lists/visda/validation_imagelist.txt'
		else: 
			folder = './data/'
			args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
			args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
			args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

		if args.dset == 'office-home':
			if args.da == 'pda':
				args.class_num = 65
				args.src_classes = [i for i in range(65)]
				args.tar_classes = [i for i in range(25)]

		if not os.path.exists(args.output):
			os.mkdir(args.output)

		args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
		args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
		args.name = names[args.s][0].upper()+names[args.t][0].upper()

		if not osp.exists(args.output_dir):
			os.system('mkdir -p ' + args.output_dir)
		if not osp.exists(args.output_dir):
			os.mkdir(args.output_dir)

		args.savename = 'par_' + str(args.cls_par)
		if args.da == 'pda':
			args.gent = ''
			args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
		args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
		args.out_file.write(print_args(args)+'\n')
		args.out_file.flush()
		train_target(args)
		
		test_target(args)