# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory,ClusterMemory_all
from clustercontrast.trainers import ClusterContrastTrainer_pretrain_camera_interC,ClusterContrastTrainer_pretrain_camera_interM,ClusterContrastTrainer_pretrain_joint
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam,MoreCameraSampler
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter
start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset






def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):




    # train_transformer = T.Compose([
    #     T.Resize((height, width), interpolation=3),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop((height, width)),
    #     T.ToTensor(),
    #     normalizer,
    #     T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    # ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=False, drop_last=True), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):




    # train_transformer = T.Compose([
    #     T.Resize((height, width), interpolation=3),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop((height, width)),
    #     T.ToTensor(),
    #     normalizer,
    #     T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    # ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=False, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=False, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=2,
        shuffle=False, pin_memory=False)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input,input, 1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc
    
def extract_query_feat(model,query_loader,nquery):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input, input,2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP

def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()

def select_merge_data(u_feas, label, label_to_images,  ratio_n,  dists,rgb_num,ir_num):

    dists = torch.from_numpy(dists)
    # homo_mask = torch.zeros(len(u_feas), len(u_feas))
    # homo_mask[:rgb_num,:rgb_num] = 9900000 #100000
    # homo_mask[rgb_num:,rgb_num:] = 9900000
    # homo_mask[rgb_num:,:rgb_num] = 9900000
    print(dists.size())
    # dists.add_(torch.tril(900000 * torch.ones(len(u_feas), len(u_feas))))
    # print(dists.size())
    # dists.add_(homo_mask)
    # cnt = torch.FloatTensor([ len(label_to_images[label[idx]]) for idx in range(len(u_feas))])
    # dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))
    
    # for idx in range(len(u_feas)):
    #     for j in range(idx + 1, len(u_feas)):
    #         if label[idx] == label[j]:
    #             dists[idx, j] = 900000
    # print('rgb_num',rgb_num)
    # print('ir_num',ir_num)
    dists = dists.numpy()

    # dists=dists[:rgb_num,rgb_num:]
    ind = np.unravel_index(np.argsort(dists, axis=None)[::-1], dists.shape) #np.argsort(dists, axis=1)#
    idx1 = ind[0]
    idx2 = ind[1]
    dist_list = dists[idx1,idx2] #[dists[i,j] for i,j in zip(idx1,idx2)]
    # print(ind.shape)
    # print(ind)
    return idx1, idx2, dist_list

def select_merge_data_jacard(u_feas, label, label_to_images,  ratio_n,  dists,rgb_num,ir_num):

    dists = torch.from_numpy(dists)

    print(dists.size())

    dists = dists.numpy()

    ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
    idx1 = ind[0]
    idx2 = ind[1]
    dist_list = dists[idx1,idx2] 
    return idx1, idx2, dist_list


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



def main_worker(args):
    @torch.no_grad()
    def generate_cluster_features(labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers
    global start_epoch, best_mAP

    model = create_model(args)


    args.logs_dir = osp.join(args.logs_dir+'/'+'sysu_val2')
    
    start_time = time.monotonic()
    # cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))#str(start_epoch)+
    print("==========\nArgs:{}\n==========".format(args))
    # print('start_epoch',start_epoch)

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model

    # Evaluator
    evaluator = Evaluator(model)
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    ####################
    # if start_epoch != 0:
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    # print('start lr:',optimizer.state_dict()['param_groups'][0]['lr'])
    ####################
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    height=args.height
    width=args.width
    train_transformer_rgb = T.Compose([
    T.Resize((height, width), interpolation=3),
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5)
    ])
    
    train_transformer_rgb1 = T.Compose([
    T.Resize((height, width), interpolation=3),
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5),
    ChannelExchange(gray = 2)
    # Gray()
    ])

    transform_thermal = T.Compose( [
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((288, 144)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)
        ])
    transform_thermal1 = T.Compose( [
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((288, 144)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)])

    trainer_intrac = ClusterContrastTrainer_pretrain_joint(model)
    trainer_interc = ClusterContrastTrainer_pretrain_camera_interC(model) 
    trainer_interm = ClusterContrastTrainer_pretrain_camera_interM(model)
    stage_cmass = 30
    for epoch in range(start_epoch,args.epochs):
        print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])

        rgb_cams = np.unique([x[2] for x in dataset_rgb.train])
        # if epoch < 1:
        #     loop_iter = 5
        # else:
        loop_iter = 1
        for rgb_cam_id in sorted(rgb_cams):
            ir_cams = np.unique([x[2] for x in dataset_ir.train])
            for ir_cam_id in sorted(ir_cams):
                print('==> Create pseudo labels for camera unlabeled RGB data')
                # print('rgb, camera ',rgb_cam_id)
                cam_data_rgb = [x for x in dataset_rgb.train if x[2] == rgb_cam_id]
                cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                 64, args.workers, 
                                                 testset=sorted(cam_data_rgb))
                features_rgb_dict, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
                features_rgb = torch.cat([features_rgb_dict[f].unsqueeze(0) for f, _, _ in sorted(cam_data_rgb)], 0)
                rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)
                rgb_eps= 0.55#0.55
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
                cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
                pseudo_labeled_dataset_rgb = []
                modality_rgb = []
                cams_rgb=[]
                for i, ((fname, _, cid), label) in enumerate(zip(sorted(cam_data_rgb), pseudo_labels_rgb)):
                    cams_rgb.append(cid)
                    modality_rgb.append(0)
                    if label != -1:
                        pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))

                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, use_hard=args.use_hard).cuda()
                memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
                print('==> Statistics for RGB  epoch {}: camera {} {} clusters,cluster data {} '.format(epoch,rgb_cam_id, num_cluster_rgb,len(pseudo_labeled_dataset_rgb)))

                print('==> Create pseudo labels for camera unlabeled ir data ')
                # print('ir, camera ',ir_cam_id)
                cam_data_ir = [x for x in dataset_ir.train if x[2] == ir_cam_id]
                cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                 64, args.workers, 
                                                 testset=sorted(cam_data_ir))
                features_ir_dict, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
                features_ir = torch.cat([features_ir_dict[f].unsqueeze(0) for f, _, _ in sorted(cam_data_ir)], 0)
                rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)
                ir_eps = 0.55 #0.55
                print('ir Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
                cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
                pseudo_labeled_dataset_ir = []
                modality_ir = []
                cams_ir=[]
                for i, ((fname, _, cid), label) in enumerate(zip(sorted(cam_data_ir), pseudo_labels_ir)):
                    cams_ir.append(cid)
                    modality_ir.append(1)
                    if label != -1:
                        pseudo_labeled_dataset_ir.append((fname, label.item(), cid))

                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                                   momentum=args.momentum, use_hard=args.use_hard).cuda()
                memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
                print('==> Statistics for RGB  epoch {}: camera {} {} clusters,cluster data {} '.format(epoch,rgb_cam_id, num_cluster_rgb,len(pseudo_labeled_dataset_rgb)))
                print('==> Statistics for ir  epoch {}: camera {} {} clusters,cluster data {} '.format(epoch,ir_cam_id, num_cluster_ir,len(pseudo_labeled_dataset_ir)))
                train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                            256, args.workers, args.num_instances, 50,
                                            trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)
                train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        128, args.workers, args.num_instances, 50,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

                trainer_intrac.memory_ir = memory_ir
                trainer_intrac.memory_rgb = memory_rgb
                train_loader_ir.new_epoch()
                time.sleep(1)
                train_loader_rgb.new_epoch()
                time.sleep(1)
                trainer_intrac.train(epoch, train_loader_ir,train_loader_rgb, optimizer, print_freq=10, train_iters=len(train_loader_rgb))

                if ir_cam_id == 0:
                    ir_softmax_dim=[]
                    distribute_map_ir = F.normalize(memory_ir.features.data)
                    ir_softmax_dim.append(distribute_map_ir.size(0))
                else:
                    distribute_tmp = F.normalize(memory_ir.features.data)
                    distribute_map_ir = torch.cat((distribute_map_ir, distribute_tmp), dim=0)
                    ir_softmax_dim.append(distribute_map_ir.size(0))
                del train_loader_ir, train_loader_rgb
            if rgb_cam_id == 0:
                rgb_softmax_dim=[]
                distribute_map_rgb = F.normalize(memory_rgb.features.data)
                rgb_softmax_dim.append(distribute_map_rgb.size(0))
            else:
                distribute_tmp = F.normalize(memory_rgb.features.data)
                distribute_map_rgb = torch.cat((distribute_map_rgb, distribute_tmp), dim=0)
                rgb_softmax_dim.append( distribute_map_rgb.size(0))
        print('distribute_map_rgb',distribute_map_rgb.size(0))
        print('distribute_map_ir',distribute_map_ir.size(0))
        model.module.classifier_rgb = nn.Linear(2048, distribute_map_rgb.size(0), bias=False).cuda()
        model.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())

        model.module.classifier_ir = nn.Linear(2048, distribute_map_ir.size(0), bias=False).cuda()
        model.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())
#############stage2
        print('==> Create pseudo labels stage2 for all unlabeled RGB data')
        cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                         256, args.workers, 
                                         testset=sorted(dataset_rgb.train))
        features_rgb_dict, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
        features_rgb_ = torch.cat([features_rgb_dict[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
        print(features_rgb_.size())
        print(distribute_map_rgb.size())
        features_rgb = F.normalize(features_rgb_).cuda()#features_rgb_.cuda()#
        features_rgb = model.module.classifier_rgb(features_rgb)*20
        print('rgb_softmax_dim',rgb_softmax_dim)
        features_rgb_1 = F.softmax(features_rgb[:,:rgb_softmax_dim[0]], dim=1)
        features_rgb_2 = F.softmax(features_rgb[:,rgb_softmax_dim[0]:rgb_softmax_dim[1]+rgb_softmax_dim[0]], dim=1)
        features_rgb_3 = F.softmax(features_rgb[:,rgb_softmax_dim[1]+rgb_softmax_dim[0]:rgb_softmax_dim[1]+rgb_softmax_dim[0]+rgb_softmax_dim[2]], dim=1)
        features_rgb_4 = F.softmax(features_rgb[:,rgb_softmax_dim[1]+rgb_softmax_dim[0]+rgb_softmax_dim[2]:], dim=1)
        features_rgb = torch.cat((features_rgb_1,features_rgb_2,features_rgb_3,features_rgb_4), dim=1)



        features_rgb = features_rgb.cpu().detach().data#
        rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)
        rgb_eps=0.6
        print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
        cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)
        pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb_)

        num_cluster_rgb_ori = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)



        print('==> Create pseudo labels stage2 for all unlabeled ir data')

        cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                         256, args.workers, 
                                         testset=sorted(dataset_ir.train))
        features_ir_dict, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
        features_ir_ = torch.cat([features_ir_dict[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

        features_ir =  F.normalize(features_ir_).cuda()#features_ir_.cuda()#
        features_ir = model.module.classifier_ir(features_ir)*20

        print('ir_softmax_dim',ir_softmax_dim)
        features_ir_1 = F.softmax(features_ir[:,:ir_softmax_dim[0]], dim=1)
        features_ir_2 = F.softmax(features_ir[:,ir_softmax_dim[0]:], dim=1)
        features_ir = torch.cat((features_ir_1,features_ir_2), dim=1)


        features_ir = features_ir.cpu().detach().data#
        # features_ir = F.softmax(features_ir, dim=1)

        # features_ir = F.normalize(features_ir) 
        rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)
        ir_eps = 0.6
        print('ir Clustering criterion: eps: {:.3f}'.format(ir_eps))
        cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
        pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
#################

################################ IR2RGB
        
        if epoch >= stage_cmass:
            lp_feat_rgb = features_rgb_.cpu().data#generate_cluster_features(pseudo_labels_rgb, features_rgb)
            lp_feat_ir =  features_ir_.cpu().data#generate_cluster_features(pseudo_labels_ir, features_ir)

            if epoch  < 100:
                W       = torch.mm(F.normalize(lp_feat_rgb),F.normalize(lp_feat_ir).t())
                W       = F.softmax(W, dim=1)
                N = W.size(0)
                topk, indices = torch.topk(W, 20)#20
                mask = torch.zeros_like(W)
                mask = mask.scatter(1, indices, 1)
                W    = W*mask
                S    = W#D1*W*D2
            else:
                rerank_dist_lp = compute_jaccard_distance_cm(lp_feat_rgb,lp_feat_ir, k1=20, k2=args.k2,search_option=3)

                rerank_dist_lp = torch.from_numpy(rerank_dist_lp)

                W = rerank_dist_lp#F.softmax(rerank_dist_lp, dim=1)
                topk, indices = torch.topk(W, 20)#20
                mask = torch.zeros_like(W)
                mask = mask.scatter(1, indices, 1)
                W    = W*mask
                S    = W#D1*W*D2
            labels = torch.from_numpy(pseudo_labels_ir+1).view(-1)
            # print(labels.max())
            # print(c)
            c = int(num_cluster_ir)+1
            n = labels.size(0)
            # print('labels.size()',labels.size())
            y = F.one_hot(labels.view(n,1).long(),c).float().squeeze(1) 
            
            adj = S
            # print(adj)
            post_step=lambda x:torch.clamp(x,0,1)
            result = y.clone()
            alpha = 1
            result_soft = alpha * (adj @ result)
            # print(result_soft.size())
            result = torch.zeros(result_soft.size(0)).view(-1)
            for i in range(result_soft.size(0)):
                # print(result_soft[i])
                zero_num = result_soft[i].sum()
                # print(zero_num)
                if zero_num == 0:
                    # print(zero_num)
                    result[i]=0
                else:
                    result[i] = torch.argmax(result_soft[i,:]).view(-1)


            print(result)
            pseudo_labels_rgb_cm =result.numpy() 
            pseudo_labels_rgb_cm = [int(i-1) for i in pseudo_labels_rgb_cm]
            pseudo_labels_rgb_cm = np.array(pseudo_labels_rgb_cm)

    ###########structure smooth soft
            if (epoch >= (stage_cmass+10)):# and (epoch %2 == 0):#5

                print('soft structure smooth')
                rgb_cm_label = torch.from_numpy(pseudo_labels_rgb_cm+1).view(-1)
                rgb_cm_label = F.one_hot(rgb_cm_label.view(lp_feat_rgb.size(0),1).long(),int(num_cluster_ir)+1).float().squeeze(1) 

                rgb_self_sim = torch.mm(F.normalize(lp_feat_rgb),F.normalize(lp_feat_rgb).t())
                topk_self, indices_self = torch.topk(rgb_self_sim, 20)#20
                mask_self = torch.zeros_like(rgb_self_sim)
                mask_self = mask_self.scatter(1, indices_self, 1)
                # mask_self = rgb_self_sim.ge(0.3).int()
                rgb_self_sim    = mask_self#rgb_self_sim*mask_self

                smooth_rgb = torch.mm(rgb_self_sim,rgb_cm_label)
                smooth_rgb = torch.argmax(smooth_rgb,1).view(-1).numpy()
                pseudo_labels_rgb_cm = [int(i-1) for i in smooth_rgb]
                pseudo_labels_rgb_cm = np.array(pseudo_labels_rgb_cm)


            pseudo_labels_ir2rgb = np.hstack((pseudo_labels_rgb_cm, pseudo_labels_ir))

            pseudo_labels_all=pseudo_labels_ir2rgb

            
            pseudo_labels_all = np.array(pseudo_labels_all)
            pseudo_labels_rgb=pseudo_labels_all[:features_rgb_.size(0)]
            pseudo_labels_ir=pseudo_labels_all[features_rgb_.size(0):]
            pseudo_labels_rgb = np.array(pseudo_labels_rgb)
            pseudo_labels_ir = np.array(pseudo_labels_ir)



        
        pseudo_labeled_dataset_rgb = []
        modality_rgb = []
        cams_rgb=[]
        print('LP rgb label')
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            
            cams_rgb.append(cid)
            modality_rgb.append(0)
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))

        num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb_ori, temp=args.temp,
                           momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        distribute_cm_map_rgb = memory_rgb.features.data

        #     distribute_tmp = memory_rgb.features.data
        
        print('==> Statistics for RGB  epoch {}:  {} clusters,cluster data {}'.format(epoch, num_cluster_rgb,len(pseudo_labeled_dataset_rgb)))
        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        128, args.workers, args.num_instances, 200,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)


        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir_)
        pseudo_labeled_dataset_ir = []
        modality_ir = []
        cams_ir=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            cams_ir.append(cid)
            modality_ir.append(1)
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))

        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                           momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()

        distribute_cm_map_ir = memory_ir.features.data#cluster_features_ir#

        print('==> Statistics for ir  epoch {}: {} clusters,cluster data {}'.format(epoch, num_cluster_ir,len(pseudo_labeled_dataset_ir)))
        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                    256, args.workers, args.num_instances, 200,
                                    trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)
        if epoch >= stage_cmass:
            trainer_interc.memory_ir = memory_ir
            trainer_interc.memory_rgb = memory_ir
        else:
            trainer_interc.memory_ir = memory_ir
            trainer_interc.memory_rgb = memory_rgb

        cams_rgb = np.asarray(cams_rgb)
        cams_ir = np.asarray(cams_ir)
        modality_rgb = np.asarray(modality_rgb)
        modality_ir = np.asarray(modality_ir)
        intra_id_features_rgb,intra_id_labels_rgb = [],[]
        intra_id_features_ir,intra_id_labels_ir = [],[]


        del features_ir,features_rgb,features_ir_,features_rgb_
        train_loader_ir.new_epoch()
        time.sleep(1)
        train_loader_rgb.new_epoch()
        time.sleep(1)
        trainer_interc.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
            intra_id_labels_rgb=intra_id_labels_rgb, intra_id_features_rgb=intra_id_features_rgb,intra_id_labels_ir=intra_id_labels_ir, intra_id_features_ir=intra_id_features_ir,
            all_label_rgb=pseudo_labels_rgb,all_label_ir=pseudo_labels_ir,cams_ir=cams_ir,cams_rgb=cams_rgb,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        ############stage3
        if epoch < (stage_cmass+10):
            distribute_cm_map = torch.cat((distribute_cm_map_rgb, distribute_cm_map_ir), dim=0)
            distribute_cm_map = F.normalize(distribute_cm_map) 
            model.module.classifier_rgb = nn.Linear(2048, distribute_cm_map.size(0), bias=False).cuda()
            model.module.classifier_rgb.weight.data.copy_(distribute_cm_map.cuda())

            model.module.classifier_ir = nn.Linear(2048, distribute_cm_map.size(0), bias=False).cuda()
            model.module.classifier_ir.weight.data.copy_(distribute_cm_map.cuda())

            print('==> Create pseudo labels for stage3 all unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb_dict, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            features_rgb_ = torch.cat([features_rgb_dict[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            print(features_rgb_.size())
            print(distribute_cm_map.size())
            features_rgb = F.normalize(features_rgb_).cuda()#features_rgb_.cuda()#
            features_rgb = model.module.classifier_rgb(features_rgb)*20
            features_rgb_1 = F.softmax(features_rgb[:,:distribute_cm_map_rgb.size(0)].data, dim=1)
            features_rgb_2 = F.softmax(features_rgb[:,distribute_cm_map_rgb.size(0):].data, dim=1)
            features_rgb = torch.cat((features_rgb_1,features_rgb_2), 1)

            features_rgb = features_rgb.cpu().detach().data#
            # features_rgb = F.softmax(features_rgb.data, dim=1)

            print('==> Create pseudo labels for all unlabeled ir data')

            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir_dict, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            features_ir_ = torch.cat([features_ir_dict[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            features_ir = F.normalize(features_ir_).cuda()#features_ir_.cuda()#
            features_ir = model.module.classifier_ir(features_ir)*20
            features_ir_1 = F.softmax(features_ir[:,:distribute_cm_map_rgb.size(0)].data, dim=1)
            features_ir_2 = F.softmax(features_ir[:,distribute_cm_map_rgb.size(0):].data, dim=1)
            features_ir = torch.cat((features_ir_1,features_ir_2), 1)

            features_ir = features_ir.cpu().detach().data#



            features_all = torch.cat((features_rgb,features_ir),dim=0)

            features_all_ = torch.cat((features_rgb_,features_ir_),dim=0)

            rerank_dist_all = compute_jaccard_distance(features_all, k1=args.k1, k2=args.k2,search_option=3)
            all_eps=0.6
            print('all Clustering criterion: eps: {:.3f}'.format(all_eps))
            cluster_all = DBSCAN(eps=all_eps, min_samples=4, metric='precomputed', n_jobs=-1)
            pseudo_labels_all = cluster_all.fit_predict(rerank_dist_all)


            pseudo_labels_rgb=pseudo_labels_all[:features_rgb_.size(0)]
            pseudo_labels_ir=pseudo_labels_all[features_rgb_.size(0):]

            cluster_features_all = generate_cluster_features(pseudo_labels_all, features_all_)
            pseudo_labeled_dataset_rgb = []
            modality_rgb = []
            cams_rgb=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
                cams_rgb.append(cid)
                modality_rgb.append(0)
                if label != -1:
                    pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))


            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)


            print('==> Statistics for RGB  epoch {}:  {} clusters,cluster data {}'.format(epoch, num_cluster_rgb,len(pseudo_labeled_dataset_rgb)))
            train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            128, args.workers, args.num_instances, 200,
                                            trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

            # cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='euclidean', n_jobs=-1)
            # pseudo_labels_ir = cluster_rgb.fit_predict(features_ir.cpu())

            pseudo_labeled_dataset_ir = []
            modality_ir = []
            cams_ir=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
                cams_ir.append(cid)
                modality_ir.append(1)
                if label != -1:
                    pseudo_labeled_dataset_ir.append((fname, label.item(), cid))

            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)


            print('==> Statistics for ir  epoch {}: {} clusters,cluster data {}'.format(epoch, num_cluster_ir,len(pseudo_labeled_dataset_ir)))
            train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        256, args.workers, args.num_instances, 200,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

            num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)
            
            memory_all = ClusterMemory(model.module.num_features, num_cluster_all, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_all.features = F.normalize(cluster_features_all, dim=1).cuda()

            trainer_interm.memory_ir = memory_all
            trainer_interm.memory_rgb = memory_all

            cams_rgb = np.asarray(cams_rgb)
            cams_ir = np.asarray(cams_ir)

            intra_id_features_rgb,intra_id_labels_rgb = [],[]
            intra_id_features_ir,intra_id_labels_ir = [],[]


            del features_ir,features_rgb,features_ir_,features_rgb_
            train_loader_ir.new_epoch()
            time.sleep(1)
            train_loader_rgb.new_epoch()

            time.sleep(1)
            trainer_interm.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                intra_id_labels_rgb=intra_id_labels_rgb, intra_id_features_rgb=intra_id_features_rgb,intra_id_labels_ir=intra_id_labels_ir, intra_id_features_ir=intra_id_features_ir,
                all_label_rgb=pseudo_labels_rgb,all_label_ir=pseudo_labels_ir,cams_ir=cams_ir,cams_rgb=cams_rgb,
                          print_freq=args.print_freq, train_iters=len(train_loader_ir))

###################


        if epoch>=6 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
            _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path='/dat01/chenjun3/data/sysu'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(1):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            cmc = all_cmc / 1
            mAP = all_mAP / 1
            mINP = all_mINP / 1
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
#################################
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_mAP': best_mAP,
                'optimizer':optimizer.state_dict()
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))

    size_ir = checkpoint['state_dict']['module.classifier_ir.weight'].size(0)
    size_rgb = checkpoint['state_dict']['module.classifier_rgb.weight'].size(0)
    model.module.classifier_ir = nn.Linear(2048, size_ir, bias=False).cuda()
    model.module.classifier_rgb = nn.Linear(2048, size_rgb, bias=False).cuda()

    model.load_state_dict(checkpoint['state_dict'])
    _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
    _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
    mode='all'
    data_path='/dat01/chenjun3/data/sysu'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=3)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=288, help="input height")
    parser.add_argument('--width', type=int, default=144, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20,40],
                        help='milestones for the learning rate decay')
    main()
