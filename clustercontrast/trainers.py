from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F





def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class ClusterContrastTrainer_pretrain_joint(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_joint, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)


            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir+loss_rgb# + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)



class ClusterContrastTrainer_pretrain_camera_interC(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_interC, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        # ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        # concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,cids_rgb,cids_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,cid_rgb=cids_rgb,cid_ir=cids_ir)
            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()


################camera

            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)
  
##################
            lamda_c = 0.1
            loss = loss_ir+loss_rgb#+lamda_c*(loss_camera_ir+loss_camera_rgb) #+ loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir)


class ClusterContrastTrainer_pretrain_camera_interM(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_interM, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()


        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)

            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,cids_rgb,cids_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,cid_rgb=cids_rgb,cid_ir=cids_ir)

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 

            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)



            loss = loss_ir+loss_rgb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'

                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir)








