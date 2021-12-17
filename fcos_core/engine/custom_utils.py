
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def read_json(json_file):
    with open(json_file, 'r') as f:
        return json.loads(f.read())

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def op(loss_dis, dis_func):
    if dis_func == "sum":
        return loss_dis
    elif dis_func == "mean":
        return loss_dis / 5
    else:
        raise NotImplementedError

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, grad_in, grad_out):
        grad_weight = grad_in[0].clone().detach().squeeze()
        self.grad_weight = grad_weight
        self.grad_out = grad_out
    def close(self):
        self.hook.remove()

def get_size(image_size):
    min_size = (512,)
    max_size = 800
    w, h = image_size
    size = random.choice(min_size)
    max_size = max_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)

def plot(name, tensor):
    plt.plot(tensor.cpu().numpy())
    plt.savefig(name)
    print("==> Saved figure to ", name)
    plt.close()