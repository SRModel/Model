import torch.nn as nn
import torch.nn.functional as F
import torch
import basicsr


import functools
# from basicsr.archs import block as B

import basicsr.archs.arch_util
from basicsr.archs.arch_util import default_init_weights

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    return nn.Sequential(#400epoch 32.726 28.623
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1),
                           bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), (3, 1), 1, (1, 0), bias=True),
        nn.Conv2d(int(in_channels * 0.5), out_channels, 1, 1, bias=True)
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)






def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride) #Conv2d(50, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    pixel_shuffle = nn.PixelShuffle(upscale_factor) #PixelShuffle(upscale_factor=4)
    return sequential(conv, pixel_shuffle)








class AAN(nn.Module):
    def __init__(self,nf,nb):
        super(AAN, self).__init__()

        AAB_block_f = functools.partial(AAB, nf=nf)

        self.conv_first = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.AAB_trunk = make_layer(AAB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.AAB_trunk(fea))
        fea = fea - trunk
        out = self.conv_last(fea)
        return out





def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class PA(nn.Module):

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out

class AttentionBranch(nn.Module):
    def __init__(self, nf, k_size=3):
        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out





def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res





class FEM(nn.Module):


    def __init__(self, num_feat, num_grow_ch=32):
        super(FEM, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb4 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.rdb4(out)
        return out * 0.2 + x

class ResidualDenseBlock(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x

class FE_Block(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(FE_Block, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)





        basicsr.archs.arch_util.default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        out = x5 + x
        return out




class PIIB(nn.Module):


    def __init__(self, num_feat=64):
        super(PIIB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.nearest = nn.UpsamplingNearest2d(scale_factor=2)
        self.t = 30
        self.K = 2
        self.reduction = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.Dynamic_adjustment = nn.Sequential(
            nn.Linear(num_feat, num_feat // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_feat // self.reduction, self.K, bias=False),
        )

        self.scaling_factor = 0.1
    def forward(self, x):
        a, b, c, d = x.shape
        original = self.conv1(x)

        y = self.avg_pool(x).view(a, b)
        y = self.Dynamic_adjustment(y)
        ax = F.softmax(y / self.t, dim=1)
        out = F.avg_pool2d(original, kernel_size=2, stride=2)
        FL = self.nearest(out)
        FL = F.interpolate(FL, size=original.shape[2:])
        FH = original - FL

        B, C, H, W = FH.shape
        FH_cp = FH.view(C, -1)

        mean_FH = torch.mean(FH_cp, dim=1)
        std_FH = torch.std(FH_cp, dim=1)

        mu_hat_FH = mean_FH.mean()
        sigma_hat_FH = mean_FH.std()
        mu_tilde_FH = std_FH.mean()
        sigma_tilde_FH = std_FH.std()

        mu_new = torch.normal(mu_hat_FH.view(1, 1).repeat(B, C),
                              self.scaling_factor * sigma_hat_FH.view(1, 1).repeat(B, C))
        sigma_new = torch.normal(mu_tilde_FH.view(1, 1).repeat(B, C),
                                 self.scaling_factor * sigma_tilde_FH.view(1, 1).repeat(B, C))

        mu_new_reshape = mu_new.view(B, C, 1, 1).repeat(1, 1, H, W)
        sigma_new_reshape = sigma_new.view(B, C, 1, 1).repeat(1, 1, H, W)

        normalized_FH = (FH - mean_FH.view(1, C, 1, 1).repeat(B, 1, H, W)) / \
                        std_FH.view(1, C, 1, 1).repeat(B, 1, H, W)

        NewStyle = sigma_new_reshape * normalized_FH + mu_new_reshape

        StylizedFeature = NewStyle * ax[:, 0].view(a, 1, 1, 1) + FL * ax[:, 1].view(a, 1, 1, 1)
        return StylizedFeature




class RDB(nn.Module):

    def __init__(self, num_feat, num_grow_ch=32):
        super(RDB, self).__init__()
        self.FEN1 = FE_Block(num_feat, num_grow_ch)
        self.FEN2 = FE_Block(num_feat, num_grow_ch)
        self.FEN3 = FE_Block(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.FEN1(x)
        out = self.FEN2(out)
        out = self.FEN3(out)

        return out + x

class RDB_EDGE(nn.Module):

    def __init__(self, num_feat, num_grow_ch=32):
        super(RDB_EDGE, self).__init__()
        self.RDB_EDGE1 = RDB_EDGE_Block(num_feat, num_grow_ch)
        self.RDB_EDGE2 = RDB_EDGE_Block(num_feat, num_grow_ch)
        self.RDB_EDGE3 = RDB_EDGE_Block(num_feat, num_grow_ch)
        self.RDB_EDGE4 = RDB_EDGE_Block(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.RDB_EDGE1(x)
        out = self.RDB_EDGE2(out)
        out = self.RDB_EDGE3(out)
        out = self.RDB_EDGE4(out)

        return out + x


class RDB_EDGE_Block(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(RDB_EDGE_Block, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        basicsr.archs.arch_util.default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        out = x5 + x
        return out



class Edge_NetV2(nn.Module):
    def __init__(self, num_feat, res_scale=1):
        super(Edge_NetV2, self).__init__()
        self.res_scale = res_scale
        self.concat_dim_reduction = nn.Conv2d(6 * num_feat, num_feat, 1, 1, 0, bias=True)
        self.RDB_EDGE_4 = RDB_EDGE(num_feat)
        self.conv_1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

    def forward(self, x):

        concatenated_features = torch.cat(x, dim=1)
        out_fea = self.concat_dim_reduction(concatenated_features)
        identity = out_fea
        out = self.conv_1(out_fea)
        out = self.RDB_EDGE_4(out)
        out = self.conv_2(out)
        return identity - out * self.res_scale



