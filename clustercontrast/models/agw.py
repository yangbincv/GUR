import torch
import torch.nn as nn
from torch.nn import init
from .resnet_agw import resnet50 as resnet50_agw
import collections
from torch.nn import functional as F
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50_agw(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        # self.AIBN =  AIBNorm2d(64,
        #                   adaptive_weight=None, generate_weight=True)
    def forward(self, x):
        x = self.visible.conv1(x)
        # x = self.AIBN(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50_agw(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        # self.AIBN =  AIBNorm2d(64,
        #                   adaptive_weight=None, generate_weight=True)
    def forward(self, x):
        x = self.thermal.conv1(x)
        # x = self.AIBN(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50_agw(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        # self.MAM3 = MAM(1024)
        # self.MAM4 = MAM(2048)
    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        # x = self.MAM3(x)
        x = self.base.layer4(x)
        # x = self.MAM4(x)
        return x

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        planes=2048
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm1d(half1, affine=True)
        self.BN = nn.BatchNorm1d(half1)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].view(-1,self.half,1).contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1.view(-1,self.half).contiguous(), out2), 1)
        return out

class BN_weight(nn.Module):
    def __init__(self, planes):
        super(BN_weight, self).__init__()
        planes=2048
        dim = planes
        r =16
        self.IN = nn.InstanceNorm1d(planes, affine=True)
        self.BN = nn.BatchNorm1d(planes)
        self.channel_attention_in = nn.Sequential(
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        # self.channel_attention_bn = nn.Sequential(
        #         nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
        #         nn.Sigmoid()
        #     )
    def forward(self, x):
        b, c  = x.shape
        x = x.view(b,c,1,1)
        mask_b = self.channel_attention_in(x).contiguous().view(-1,2048)
        # print('mask_i.shape',mask_i.shape)
        x = x.view(b,c)
        # out_in = self.IN(x.view(-1,2048,1).contiguous()).view(-1,2048)
        out_bn = self.BN(x.contiguous())
        out = out_bn * mask_b + out_bn * (1 - mask_b)
        return out

class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()
        
        self.channel_attention_in = nn.Sequential(
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.channel_attention_bn = nn.Sequential(
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.IN = nn.InstanceNorm2d(dim, affine=True)
        self.BN = nn.BatchNorm2d(dim)
        # self.softmax = nn.Softmax(dim=1)
        # self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)
    def forward(self, x):
        b, c, h, w = x.shape
        # x_inter = x.view(b, c, -1)
        # p = 1.0
        # pooled = (torch.mean(x_inter**p, dim=-1) + 1e-12)**(1/p)
        # pooled = pooled.view(b,c,1,1)
        # pooled = F.avg_pool2d(x, x.size()[2:])
        # mask_i = self.channel_attention_in(pooled)
        # x = x * mask_i + self.IN(x) * (1 - mask_i)#+
        # x = self.IN(x) * mask_i + self.BN(x) * mask_b+(1-mask_i)*x+(1-mask_b)*x
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask_b = self.channel_attention_bn(pooled)
        x = self.BN(x) * mask_b+(1-mask_b)*x
        return mask_b

class AIBNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, only_bn=False,
                 last_gamma=False, adaptive_weight=None, generate_weight=False):
        super(AIBNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.only_bn = only_bn
        self.last_gamma = last_gamma
        self.generate_weight = generate_weight
        if generate_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        if not only_bn:
            if adaptive_weight is not None:
                self.adaptive_weight = adaptive_weight
            else:
                self.adaptive_weight = nn.Parameter(torch.ones(1) * 0.1)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):

        self.running_mean.zero_()
        self.running_var.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x, weight=None, bias=None):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        temp = var_in + mean_in ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_(
                    (1 - self.momentum) * mean_bn.squeeze().data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum)
                                      * var_bn.squeeze().data)
            else:
                self.running_mean.add_(mean_bn.squeeze().data)
                self.running_var.add_(
                    mean_bn.squeeze().data ** 2 + var_bn.squeeze().data)
        else:
            mean_bn = torch.autograd.Variable(
                self.running_mean).unsqueeze(0).unsqueeze(2)
            var_bn = torch.autograd.Variable(
                self.running_var).unsqueeze(0).unsqueeze(2)

        if not self.only_bn:

            adaptive_weight = torch.clamp(self.adaptive_weight, 0, 1)
            mean = (1 - adaptive_weight[0]) * \
                mean_in + adaptive_weight[0] * mean_bn
            var = (1 - adaptive_weight[0]) * \
                var_in + adaptive_weight[0] * var_bn

            x = (x-mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)
        else:
            x = (x - mean_bn) / (var_bn + self.eps).sqrt()
            x = x.view(N, C, H, W)

        if self.generate_weight:
            weight = self.weight.view(1, self.num_features, 1, 1)
            bias = self.bias.view(1, self.num_features, 1, 1)
        else:
            weight = weight.view(1, self.num_features, 1, 1)
            bias = bias.view(1, self.num_features, 1, 1)
        return x * weight + bias

#####
class embed_net_ori(nn.Module):
    def __init__(self,  num_classes=1000, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net_ori, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])
        pool_dim = 2048
        self.num_features = pool_dim
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(pool_dim, num_classes, bias=False)

        self.classifier_ir = nn.Linear(pool_dim, 1000, bias=False)
        self.classifier_rgb = nn.Linear(pool_dim, 1000, bias=False)

        self.ir_clsnum = 1000
        self.rgb_clsnum = 1000

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool
        
        # self.bn_ir = nn.BatchNorm1d(2048)
        # self.bn_rgb = nn.BatchNorm1d(2048)
        # self.bn_ir.apply(weights_init_kaiming)
        # self.bn_rgb.apply(weights_init_kaiming)
        # self.bn = BN_weight(2048)
        # self.ibn_ir = IBN_weight(2048)
        # self.ibn_rgb = IBN_weight(2048)
    def forward(self, x1, x2, modal=0,label_1=None,label_2=None,cid_rgb=None,cid_ir=None,invers_bn=False):
        # print(x1,x2)
        single_size = x1.size(0)
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            label = torch.cat((label_1, label_2), -1)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        
        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        # x_pool = self.bottleneck(x_pool)
        # feat = self.classifier_ir(x_pool)
        # x_pool = F.softmax(x_pool, dim=1)
        feat = self.bottleneck(x_pool)
        # if modal == 0:
        #     feat_rgb = self.bn_rgb(x_pool[:single_size])
        #     feat_ir = self.bn_ir(x_pool[single_size:])
        #     feat =  torch.cat((feat_rgb, feat_ir), 0)
        # if modal == 1:
        #     if invers_bn == True:
        #         feat_rgb = self.bn_ir(x_pool)
        #     else:
        #         feat_rgb = self.bn_rgb(x_pool)
        #     feat = feat_rgb
        # if modal == 2:
        #     if invers_bn == True:
        #         feat_ir = self.bn_rgb(x_pool)
        #     else:
        #         feat_ir = self.bn_ir(x_pool)
        #     feat = feat_ir
        if self.training:
            return feat,feat[:single_size],feat[single_size:],label_1,label_2,cid_rgb,cid_ir  #x_pool#, self.classifier(feat) feat_rgb,feat_ir
        else:
            # if modal == 1:
            #      return self.l2norm(feat_rgb)
            # if modal == 2:
            #      return self.l2norm(feat_ir)
            return self.l2norm(feat)#self.l2norm(x_pool)#, 

        # if self.training:
        #     return x_pool, self.classifier(feat)
        # else:
        #     return self.l2norm(x_pool), self.l2norm(feat)



def agw(pretrained=False,no_local= 'down', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = embed_net_ori(no_local= 'down', gm_pool = 'on')
    return model