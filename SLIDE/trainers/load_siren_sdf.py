import os
print(os.getcwd())
import torch
import numpy as np
import skimage.measure

from trainers import modules


class SDFDecoder(torch.nn.Module):
    def __init__(self, net_path):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type="sine", final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(net_path))
        self.model.cuda()
        



def gradient(net, vec, grad_outputs=None):
    y = net.model.forward({'coords':vec})['model_out']
    x = y['model_in']
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]

    return grad
