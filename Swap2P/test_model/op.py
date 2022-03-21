import os

import torch
from torch import nn
from torch.nn import functional as F

torch.ops.load_library('libfusebias.so')
fused = torch.ops.fusebias
torch.ops.load_library('libupfirdn.so')

upfirdn2d_op = torch.ops.upfirdn

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    empty = input.new_empty(0)
    out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)


    return out


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    
    up_x, up_y = up,up
    down_x, down_y = down,down
    pad_x0, pad_x1, pad_y0, pad_y1 = pad[0], pad[1], pad[0], pad[1]

    kernel_h, kernel_w = kernel.shape
    batch, channel, in_h, in_w = input.shape
   

    input = input.view(-1, in_h, in_w, 1)

    

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
   

    g_pad_x0 = kernel_w - pad_x0 - 1
    g_pad_y0 = kernel_h - pad_y0 - 1
    g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
    g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

   

    out = upfirdn2d_op.upfirdn2d(
        input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
    )
    # out = out.view(major, out_h, out_w, minor)
    out = out.view(-1, channel, out_h, out_w)
    return out

