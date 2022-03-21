import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
torch.ops.load_library('build/libfusebias.so')

fused = torch.ops.fusebias

#@torch.jit.script
def compute_fused(input,bias,empty,value1,value2,negative_slope,scale):
    return fused.fused_bias_act(input, bias, empty, value1, value2, negative_slope, scale)




class FuseModel(nn.Module):
    def __init__(self):
        super(FuseModel,self).__init__()
    def forward(self, input):
        bias=None
        negative_slope=0.2
        scale=2 ** 0.5
        empty = input.new_empty(0)


        if bias is None:
            bias = empty

        out = compute_fused(input, bias, empty, 3, 0, negative_slope, scale)


        return out
    




input = torch.randn(1,3,64,64).cuda()
model = FuseModel()
model.eval()
out = model(input)
#bias=None
#empty = input.new_empty(0)
#negative_slope=0.2
#scale=2 ** 0.5

#out = compute_fused(input, bias, empty, 3, 0, negative_slope, scale)
#print('out shape ',out.shape)
traced_script_module = torch.jit.trace(model, (input))
traced_script_module.save("model.pt")

