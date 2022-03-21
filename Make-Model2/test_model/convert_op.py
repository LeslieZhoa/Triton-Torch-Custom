import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
torch.ops.load_library('build/libupfirdn.so')

upfirdn2d_op = torch.ops.upfirdn

#@torch.jit.script
def compute_up(input, kernel, up, down, pad):
    up_x, up_y = up
    down_x, down_y = down
    pad_x0, pad_x1, pad_y0, pad_y1 = pad

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


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class UpModel(nn.Module):
    def __init__(self):
        super(UpModel,self).__init__()
    def forward(self, input, kernel=(1,3,3,1), up=(1,1), down=(1,1), pad=(0, 0,0,0)):
        kernel = make_kernel(kernel).to(input.device)
        out = compute_up(input, kernel, up, down, pad)

    


        return out
    




input = torch.randn(1,3,64,64).cuda()
model = UpModel()
model.eval()
out = model(input)
#bias=None
#empty = input.new_empty(0)
#negative_slope=0.2
#scale=2 ** 0.5

#out = compute_fused(input, bias, empty, 3, 0, negative_slope, scale)
print('out shape ',out.shape)
traced_script_module = torch.jit.trace(model, (input))
traced_script_module.save("model.pt")

