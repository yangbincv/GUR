#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu



@torch.no_grad()
def compute_ranked_list(features, k=20, search_option=0, fp16=False, verbose=True):

    end = time.time()
    if verbose:
        print("Computing ranked list...")

    if search_option < 3:
        torch.cuda.empty_cache()
        features = features.cuda().detach()

    ngpus = faiss.get_num_gpus()

    if search_option == 0:
        # Faiss Search + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, features, features, k+1)
        initial_rank = initial_rank.cpu().numpy()

    elif search_option == 1:
        # Faiss Search + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, features, k+1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()

    elif search_option == 2:
        # PyTorch Search + PyTorch CUDA Tensors
        torch.cuda.empty_cache()
        features = features.cuda().detach()
        dist_m = compute_euclidean_distance(features, cuda=True)
        initial_rank = torch.argsort(dist_m, dim=1)
        initial_rank = initial_rank.cpu().numpy()

    else:
        # Numpy Search (CPU)
        torch.cuda.empty_cache()
        features = features.cuda().detach()
        dist_m = compute_euclidean_distance(features, cuda=False)
        initial_rank = np.argsort(dist_m.cpu().numpy(), axis=1)
        features = features.cpu()

    features = features.cpu()
    if verbose:
        print("Ranked list computing time cost: {}".format(time.time() - end))

    return initial_rank[:, 1:k+1]


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def k_reciprocal_neigh_cm(initial_rank,initial_rank_t, i, k1):

    # print(initial_rank)
    # print(initial_rank.shape)
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank_t[forward_k_neigh_index,:k1+1]
    # backward_k_neigh_index = initial_rank_t[forward_k_neigh_index,:int((k1+1)/2)]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


def compute_jaccard_distance_cm(target_features_1,target_features_2, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')
    # target_features = torch.cat((target_features_1,target_features_2),dim=0)
    ngpus = faiss.get_num_gpus()
    # N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    # if (search_option==0):
    #     # GPU + PyTorch CUDA Tensors (1)
    #     res = faiss.StandardGpuResources()
    #     res.setDefaultNullStreamAllDevices()
    #     _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
    #     initial_rank = initial_rank.cpu().numpy()
    # elif (search_option==1):
    #     # GPU + PyTorch CUDA Tensors (2)
    #     res = faiss.StandardGpuResources()
    #     index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
    #     index.add(target_features.cpu().numpy())
    #     _, initial_rank = search_index_pytorch(index, target_features, k1)
    #     res.syncDefaultStreamCurrentDevice()
    #     initial_rank = initial_rank.cpu().numpy()
    # elif (search_option==2):
    #     # GPU
    #     index = index_init_gpu(ngpus, target_features.size(-1))
    #     index.add(target_features.cpu().numpy())
    #     _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    # else:
    #     # CPU
    #     index = index_init_cpu(target_features.size(-1))
    #     index.add(target_features.cpu().numpy())
    #     _, initial_rank = index.search(target_features.cpu().numpy(), target_features.size(0))
    W       = torch.mm(F.normalize(target_features_1),F.normalize(target_features_2).t()) #[22258,11909]
    # W       = F.softmax(W, dim=1)
    initial_rank = torch.argsort(W,dim=1,descending=True).cpu().numpy()
    initial_rank_t = torch.argsort(W.t(),dim=1,descending=True).cpu().numpy()
    # print(initial_rank)
    N = W.size(0)
    M = W.size(1)
    # print(target_features.size(0))
    # print(target_features_1.size(0))
    # print(target_features_2.size(0))
    # # print(initial_rank.shape)
    # ss_1 = np.where(initial_rank[:target_features_1.size(0),:]>=target_features_1.size(0)) #np.argwhere(initial_rank>target_features_1.size(0))# np.where(initial_rank>target_features_1.size(0))
    # ss_2 = np.where(initial_rank[target_features_1.size(0):,:]<target_features_1.size(0))
    # # ss_1 = tuple(ss_1)
    # print(ss_1[0].shape)
    # print(ss_1)
    # print(ss_2[0].shape)
    # print(ss_2)
    # # initial_rank_1 = initial_rank[:target_features_1.size(0)][ss_1].reshape(target_features_1.size(0),target_features_2.size(0))[:,:30]
    # initial_rank_1 = initial_rank[ss_1[0],ss_1[1]].reshape(target_features_1.size(0),target_features_2.size(0))[:,:30]
    # initial_rank_2 = initial_rank[ss_2[0]+target_features_1.size(0),ss_2[1]].reshape(target_features_2.size(0),target_features_1.size(0))[:,:30]
    # print(initial_rank_1.shape)
    # print(initial_rank_1)
    # print(initial_rank_2.shape)
    # print(initial_rank_2)


    
    # initial_rank = np.concatenate([initial_rank_1,initial_rank_2],axis=0)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh_cm(initial_rank,initial_rank_t, i,k1))
        # nn_k1_half.append(k_reciprocal_neigh_cm(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, M), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        # print(k_reciprocal_index)
        # if len(k_reciprocal_index) == 0:
        #     k_reciprocal_index = np.arange(M)
        k_reciprocal_expansion_index = k_reciprocal_index
        # for candidate in k_reciprocal_index:
        #     candidate_k_reciprocal_index = nn_k1_half[candidate]
        #     if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
        #         k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        # k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        if len(k_reciprocal_expansion_index) == 0:
            k_reciprocal_expansion_index = np.arange(M)
            dist = torch.zeros((1,M))
            # dist = torch.mm(F.normalize(target_features_1[i].unsqueeze(0).contiguous()), F.normalize(target_features_2[k_reciprocal_expansion_index]).t())
        else:
            dist = torch.mm(F.normalize(target_features_1[i].unsqueeze(0).contiguous()), F.normalize(target_features_2[k_reciprocal_expansion_index]).t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = dist#F.softmax(dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = dist.view(-1).cpu().numpy()#F.softmax(dist, dim=1).view(-1).cpu().numpy()

    # del nn_k1, nn_k1_half

    # # if k2 != 1:
    # #     V_qe = np.zeros_like(V, dtype=mat_type)
    # #     for i in range(N):
    # #         V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
    # #     V = V_qe
    # #     del V_qe

    # del initial_rank

    # invIndex = []
    # for i in range(N):
    #     # invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num #[0]
    #     invIndex.append(np.where(V[i,:] != 0)[0])  #len(invIndex)=all_num #[0]
    #     # print(np.where(V[:,i] != 0)[0])
    # jaccard_dist = np.zeros((N, M), dtype=mat_type)
    # for i in range(N):
    #     temp_min = np.zeros((1, M), dtype=mat_type)
    #     # temp_max = np.zeros((1,N), dtype=mat_type)
    #     indNonZero = np.where(V[i, :] != 0)[0] #[0]
    #     indImages = []
    #     indImages = [invIndex[ind] for ind in indNonZero]
    #     for j in range(len(indNonZero)):
    #         temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
    #         # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

    #     jaccard_dist[i] = 1-temp_min/(2-temp_min)
    #     # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    # # del invIndex, V

    # pos_bool = (jaccard_dist < 0)
    # jaccard_dist[pos_bool] = 0.0
    # if print_flag:
    #     print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return V#jaccard_dist

