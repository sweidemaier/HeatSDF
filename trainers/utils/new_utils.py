import numpy as np 
import torch

#TODO Florine utils sinnvoll sortieren

def tens(input_pt):
    input = np.float32(input_pt)
    output_pt = torch.tensor(input)
    output_pt = output_pt.cuda()
    output_pt.requires_grad = True
    return output_pt

#TODO Florine ?
def spherical(phi, theta):
    return 0

def comp(input_pt, count):
    if(input_pt.size(dim = 0) < count): 
        print("Error: Mismatched dimensions")
        return 0
    return input_pt[count]