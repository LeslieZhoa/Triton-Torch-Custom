from model_jiajie import Encoder, Generator
from torchvision import transforms, utils
from torch.utils import data
import   torch.nn as nn
import torch
import cv2


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


class SwapP2P(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.encoder = Encoder(channel)
        self.generator = Generator(channel)
        self.encoder.eval()
        self.generator.eval()

    def load_network(self, save_path):
        ckpt = torch.load(save_path)
        self.encoder.load_state_dict(ckpt["e_ema"])
        self.generator.load_state_dict(ckpt["g_ema"])

    def forward(self, x):
        struct, text = self.encoder(x)
        out = self.generator(struct, text)
        return out



model = SwapP2P()

model.load_network('021999.pt')
model.cuda()
img = torch.rand(1, 3, 512, 512).cuda()


m_name = "sawpae_server.pt"
with torch.jit.optimized_execution(True):
   traced_model = torch.jit.trace(model, img) 
   traced_model.save(m_name)

import numpy as np
def check_model():

    model = torch.jit.load(m_name, map_location='cuda:0')
    
    model = model.cuda(0).eval()
    img = torch.randn(1,3,512,512).cuda()
    model(img)
check_model()
# 020000.pt
