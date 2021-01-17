from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import functools

class DynamicUpsamplingFilter_3C(nn.Module):
    '''dynamic upsampling filter with 3 channels applying the same filters
    filter_size: filter size of the generated filters, shape (C, kH, kW)'''

    def __init__(self, filter_size=(1, 5, 5)):
        super(DynamicUpsamplingFilter_3C, self).__init__()
        nF = np.prod(filter_size) 
        expand_filter_np = np.reshape(np.eye(nF, nF),
                                      (nF, filter_size[0], filter_size[1], filter_size[2]))
        expand_filter = torch.from_numpy(expand_filter_np).float()
        self.expand_filter = expand_filter.repeat(128,1,1,1)  

    def forward(self, x, filters):
        '''x: input image, [B, 3, H, W]
        filters: generate dynamic filters, [B, F, R, H, W], e.g., [B, 25, 16, H, W]
            F: prod of filter kernel size, e.g., 5*5 = 25
            R: used for upsampling, similar to pixel shuffle, e.g., 4*4 = 16 for x4
        Return: filtered image, [B, 3*R, H, W]
        '''
        filters = filters.unsqueeze(dim=2)
        B, nF, R, H, W = filters.size()
        # using group convolution
        input_expand = F.conv2d(x, self.expand_filter.type_as(x), padding=1,
                                groups=128) 
        input_expand = input_expand.view(B, 128, nF, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, 128, K^2]
        filters = filters.permute(0, 3, 4, 1, 2)  # [B, H, W, K^2, 1]
        out = torch.matmul(input_expand, filters)  # [B, H, W, 128, 1]
        return out.permute(0, 3, 4, 1, 2).squeeze(dim=2)  # [B, 128, H, W]

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv_d1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_d2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv_s1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_s2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv_d1, self.conv_d2, self.conv_s1, self.conv_s2], 0.1)
    def forward(self, hd, hs):
        identity_d = hd
        out_d = F.relu(self.conv_d1(hd), inplace=True)
        out_d = self.conv_d2(out_d)

        identity_s = hs
        out_s = F.relu(self.conv_s1(hs), inplace=True)
        out_s = self.conv_s2(out_s)
     
        hsd = out_d + out_s
        return hsd + identity_d, hsd + identity_s
class neuro(nn.Module):
    def __init__(self, nf, scale):
        super(neuro,self).__init__()
        pad = (1,1)
        layers = 7
        block = []
        self.forget = nn.Conv2d(3, 3*3, 3, 1, 1)
        self.conv_d = nn.Conv2d(16*3 + nf + 3*2, nf, (3,3), stride=(1,1), padding=pad)
        self.conv_s = nn.Conv2d(16*3 + nf + 3*2, nf, (3,3), stride=(1,1), padding=pad)
        self.SD1 = ResidualBlock_noBN(nf)
        self.SD2 = ResidualBlock_noBN(nf)
        self.SD3 = ResidualBlock_noBN(nf)
        self.SD4 = ResidualBlock_noBN(nf)
        self.SD5 = ResidualBlock_noBN(nf)
        self.SD6 = ResidualBlock_noBN(nf)
        self.SD7 = ResidualBlock_noBN(nf)
        self.SD8 = ResidualBlock_noBN(nf)
        self.SD9 = ResidualBlock_noBN(nf)

        self.conv_hd = nn.Conv2d(nf, nf, (3,3), stride=(1,1), padding=pad)
        self.conv_od = nn.Conv2d(nf, 48, (3,3), stride=(1,1), padding=pad)

        self.conv_hs = nn.Conv2d(nf, nf, (3,3), stride=(1,1), padding=pad)
        self.conv_os = nn.Conv2d(nf, 48, (3,3), stride=(1,1), padding=pad)

        initialize_weights([ self.conv_d, self.conv_s, self.conv_hd, self.conv_od, self.conv_hs, self.conv_os],0.1)
        self.dyn = DynamicUpsamplingFilter_3C((1,3,3))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, D, S, hsd, od, os, ref):
        ref = self.lrelu(self.forget(ref))
        sim = F.sigmoid(self.dyn(hsd, ref))
        hsd = torch.mul(sim, hsd)

        D = torch.cat((D,hsd,od), dim=1)
        hd = self.lrelu(self.conv_d(D))
        S = torch.cat((S, hsd, os), dim=1)
        hs = self.lrelu(self.conv_s(S))

        hd, hs = self.SD1(hd, hs)
        hd, hs = self.SD2(hd, hs)
        hd, hs = self.SD3(hd, hs)
        hd, hs = self.SD4(hd, hs)
        hd, hs = self.SD5(hd, hs)
        hd, hs = self.SD6(hd, hs)
        hd, hs = self.SD7(hd, hs)
        hd, hs = self.SD8(hd, hs)
        hd, hs = self.SD9(hd, hs)

        x_hd = self.lrelu(self.conv_hd(hd))
        x_od = self.conv_od(hd)

        x_hs = self.lrelu(self.conv_hs(hs))
        x_os = self.conv_os(hs)
        return x_hd, x_hs, x_od, x_os

class RSDN9_128(nn.Module):
    def __init__(self, scale):
        super(RSDN9_128,self).__init__()
        self.nf = 128
        self.neuro = neuro(self.nf, scale)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.conv = nn.Conv2d(48*2, 48, 3, 1, 1)
        initialize_weights([self.conv],0.1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, LR, D, S):
        _,T,_,_,_ = D.shape
        assert ( T >= 3 )
        out = []
        out_d = []
        out_s = []
        init_temp = torch.zeros_like(D[:,0,0:1,:,:])
        init_od = init_temp.repeat(1,48,1,1)
        init_os = init_temp.repeat(1,48,1,1)
        init_hd = init_temp.repeat(1,self.nf,1,1)
        init_hs = init_temp.repeat(1,self.nf,1,1)
        init_D = torch.cat((D[:,1,:,:,:],D[:,0,:,:,:]),dim=1)
        init_S = torch.cat((S[:,1,:,:,:],S[:,0,:,:,:]),dim=1)
        x_hd, x_hs, x_od, x_os = self.neuro(init_D, init_S, init_hd + init_hs, init_od, init_os, LR[:,0,:,:,:])
        out_d.append(F.pixel_shuffle(x_od, self.scale))
        out_s.append(F.pixel_shuffle(x_os, self.scale))
        x_o = torch.cat((x_od, x_os),dim=1)
        x_o = self.conv(x_o)
        x_o = F.pixel_shuffle(x_o, self.scale)
        out.append(x_o)
        for i in range(T-2):
            D_ = torch.cat((D[:,i,:,:,:],D[:,i+1,:,:,:]),dim=1)
            S_ = torch.cat((S[:,i,:,:,:],S[:,i+1,:,:,:]),dim=1)
            x_hd, x_hs, x_od, x_os = self.neuro(D_, S_, x_hd+ x_hs, x_od, x_os, LR[:,i+1,:,:,:])
            out_d.append(F.pixel_shuffle(x_od, self.scale))
            out_s.append(F.pixel_shuffle(x_os, self.scale))
            x_o = torch.cat((x_od, x_os),dim=1)
            x_o = self.conv(x_o)
            x_o = F.pixel_shuffle(x_o, self.scale)
            out.append(x_o)
        D_ = torch.cat((D[:,T-2,:,:,:],D[:,T-1,:,:,:]),dim=1)
        S_ = torch.cat((S[:,T-2,:,:,:],S[:,T-1,:,:,:]),dim=1)
        x_hd, x_hs, x_od, x_os = self.neuro(D_, S_, x_hd+ x_hs,  x_od, x_os, LR[:,T-1,:,:,:])
        out_d.append(F.pixel_shuffle(x_od, self.scale))
        out_s.append(F.pixel_shuffle(x_os, self.scale))
        x_o = torch.cat((x_od, x_os),dim=1)
        x_o = self.conv(x_o)
        x_o = F.pixel_shuffle(x_o, self.scale)
        out.append(x_o)
        out = torch.stack(out,dim=1) #[B,T,C,H,W]
        out_d = torch.stack(out_d,dim=1) #[B,T,C,H,W]
        out_s = torch.stack(out_s,dim=1) #[B,T,C,H,W]
        return out, out_d, out_s

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
