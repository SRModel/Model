import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs import AttackDefensev4_block as B  
# from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer
import torchvision

 
class AttackDefensev4(nn.Module):  
    def __init__(self,
                 num_in_ch,   
                 num_out_ch,   
                 num_feat=64,   
                 upscale=4,
                 ):   
        super(AttackDefensev4, self).__init__()
        # conv = block.default_conv   
        nf = num_feat  
        self.upscale = upscale    
        self.fea_conv = B.conv_layer(num_in_ch, nf, kernel_size=3)  
        num_modules = 6  


   
        self.RDB = B.RDB(num_feat)   
        self.PIIB = B.PIIB(num_feat)   


 
        self.B1_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)   
        self.B1_s_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)   
        self.B2_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B2_s_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B3_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B3_s_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B4_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B4_s_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B5_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B5_s_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B6_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.B6_s_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积

        self.lr_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积
        self.lr_s_conv = B.conv_layer(num_feat, num_out_ch, kernel_size=1)  # 输入通道为nf通道 输出特征为3通道  1×1的卷积






        self.B1 = B.FEM(num_feat=nf)
        self.B2 = B.FEM(num_feat=nf)
        self.B3 = B.FEM(num_feat=nf)
        self.B4 = B.FEM(num_feat=nf)
        self.B5 = B.FEM(num_feat=nf)
        self.B6 = B.FEM(num_feat=nf)

        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.c_s = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf,kernel_size=3)  # Conv2d(50, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.LR_conv_s = B.conv_layer(nf, nf, kernel_size=3)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, num_out_ch, upscale_factor=upscale)

        self.edge_upsampler = upsample_block(64, 3, upscale_factor=upscale)
        self.scale_idx = 0


        self.edge_tail = B.conv_layer(nf, nf, kernel_size=3)

        self.edge = B.Edge_NetV2(nf, res_scale=1)


        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_for_out1 = nn.Linear(num_out_ch, 256)
        self.fc_for_out1_s = nn.Linear(num_out_ch, 256)

        self.fc_for_out2 = nn.Linear(num_out_ch, 256)
        self.fc_for_out2_s = nn.Linear(num_out_ch, 256)

        self.fc_for_out3 = nn.Linear(num_out_ch, 256)
        self.fc_for_out3_s = nn.Linear(num_out_ch, 256)

        self.fc_for_out4 = nn.Linear(num_out_ch, 256)
        self.fc_for_out4_s = nn.Linear(num_out_ch, 256)

        self.fc_for_out5 = nn.Linear(num_out_ch, 256)
        self.fc_for_out5_s = nn.Linear(num_out_ch, 256)

        self.fc_for_out6 = nn.Linear(num_out_ch, 256)
        self.fc_for_out6_s = nn.Linear(num_out_ch, 256)

        self.fc_for_lr = nn.Linear(num_out_ch, 256)
        self.fc_for_lr_s = nn.Linear(num_out_ch, 256)


    def forward(self, input):
        bi = F.interpolate(input, scale_factor=self.upscale, mode='bicubic', align_corners=False)  # 这里把上采样加上去了

        x = input   # 原始特征


        out_fea = self.fea_conv(input)

        RDB_fea = self.RDB(out_fea)
        PIIB_fea = self.PIIB(RDB_fea)



        out_B1 = self.B1(RDB_fea)
        out_B1_s = self.B1(PIIB_fea)

        out_b1 = self.B1_conv(out_B1)
        out_s_b1 = self.B1_s_conv(out_B1_s)



        pooled_out1 = self.adaptive_pool(out_b1)
        flattened_out1 = pooled_out1.view(pooled_out1.size(0), -1)
        output_from_out1 = self.fc_for_out1(flattened_out1)


        pooled_out1_s = self.adaptive_pool(out_s_b1)
        flattened_out1_s = pooled_out1_s.view(pooled_out1_s.size(0), -1)
        output_from_out1_s = self.fc_for_out1_s(flattened_out1_s)





        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B2_s = self.B2(out_B1_s)
        out_B3_s = self.B3(out_B2_s)
        out_B4_s = self.B4(out_B3_s)
        out_B5_s = self.B5(out_B4_s)
        out_B6_s = self.B6(out_B5_s)



        out_b2 = self.B2_conv(out_B2)
        out_s_b2 = self.B2_s_conv(out_B2_s)
        out_b3 = self.B3_conv(out_B3)
        out_s_b3 = self.B3_s_conv(out_B3_s)
        out_b4 = self.B4_conv(out_B4)
        out_s_b4 = self.B4_s_conv(out_B4_s)
        out_b5 = self.B5_conv(out_B5)
        out_s_b5 = self.B5_s_conv(out_B5_s)
        out_b6 = self.B6_conv(out_B6)
        out_s_b6 = self.B6_s_conv(out_B6_s)

        ######################################################
        pooled_out2 = self.adaptive_pool(out_b2)
        flattened_out2 = pooled_out2.view(pooled_out2.size(0), -1)
        output_from_out2 = self.fc_for_out2(flattened_out2)

        pooled_out2_s = self.adaptive_pool(out_s_b2)
        flattened_out2_s = pooled_out2_s.view(pooled_out2_s.size(0), -1)
        output_from_out2_s = self.fc_for_out2_s(flattened_out2_s)
        ######################################################
        pooled_out3 = self.adaptive_pool(out_b3)
        flattened_out3 = pooled_out3.view(pooled_out3.size(0), -1)
        output_from_out3 = self.fc_for_out3(flattened_out3)

        pooled_out3_s = self.adaptive_pool(out_s_b3)
        flattened_out3_s = pooled_out3_s.view(pooled_out3_s.size(0), -1)
        output_from_out3_s = self.fc_for_out3_s(flattened_out3_s)
        ######################################################
        pooled_out4 = self.adaptive_pool(out_b4)
        flattened_out4 = pooled_out4.view(pooled_out4.size(0), -1)
        output_from_out4 = self.fc_for_out4(flattened_out4)

        pooled_out4_s = self.adaptive_pool(out_s_b4)
        flattened_out4_s = pooled_out4_s.view(pooled_out4_s.size(0), -1)
        output_from_out4_s = self.fc_for_out4_s(flattened_out4_s)
        ######################################################
        pooled_out5 = self.adaptive_pool(out_b5)
        flattened_out5 = pooled_out5.view(pooled_out5.size(0), -1)
        output_from_out5 = self.fc_for_out5(flattened_out5)

        pooled_out5_s = self.adaptive_pool(out_s_b5)
        flattened_out5_s = pooled_out5_s.view(pooled_out5_s.size(0), -1)
        output_from_out5_s = self.fc_for_out5_s(flattened_out5_s)
        ######################################################
        pooled_out6 = self.adaptive_pool(out_b6)
        flattened_out6 = pooled_out6.view(pooled_out6.size(0), -1)
        output_from_out6 = self.fc_for_out6(flattened_out6)

        pooled_out6_s = self.adaptive_pool(out_s_b6)
        flattened_out6_s = pooled_out6_s.view(pooled_out6_s.size(0), -1)
        output_from_out6_s = self.fc_for_out6_s(flattened_out6_s)





        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_B_s = self.c_s(torch.cat([out_B1_s, out_B2_s, out_B3_s, out_B4_s, out_B5_s, out_B6_s], dim=1))

        features = [out_B1, out_B2, out_B3, out_B4, out_B5, out_B6]
        edge_pic = self.edge(features)
        edge_map = self.edge_tail(edge_pic)
        real_edge = self.edge_upsampler(edge_pic)

        out_lr = self.LR_conv(out_B) + out_fea
        out_lr_s = self.LR_conv_s(out_B_s) + out_fea

        out_b_lr = self.lr_conv(out_lr)
        out_s_b_lr = self.lr_s_conv(out_lr_s)


        # lr
        pooled_out_lr = self.adaptive_pool(out_b_lr)
        flattened_out_lr = pooled_out_lr.view(pooled_out_lr.size(0), -1)
        output_from_out_lr = self.fc_for_lr(flattened_out_lr)


        pooled_out_lr_s = self.adaptive_pool(out_s_b_lr)
        flattened_out_lr_s = pooled_out_lr_s.view(pooled_out_lr_s.size(0), -1)
        output_from_out_lr_s = self.fc_for_lr_s(flattened_out_lr_s)


        out_lr = out_lr + edge_map

        output = self.upsampler(out_lr)  # Pixel

        return output + bi, real_edge, \
               output_from_out1, output_from_out1_s, \
               output_from_out2, output_from_out2_s, \
               output_from_out3, output_from_out3_s, \
               output_from_out4, output_from_out4_s, \
               output_from_out5, output_from_out5_s, \
               output_from_out6, output_from_out6_s,\
               output_from_out_lr,output_from_out_lr_s

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx





