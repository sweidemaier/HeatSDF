import torch
import numpy as np
import torch.nn as nn


def lin_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_lin_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement
            # Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Net(nn.Module):
    def __init__(self, _, cfg):
        radius_init = 1
        super().__init__()
        self.cfg = cfg
        self.dim = dim = cfg.dim
        self.out_dim = out_dim = cfg.out_dim
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks
		
        # Network modules
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(dim, hidden_size))
        
	
        
        '''for _ in range(n_blocks):
            if _ == 0:
                lin = nn.Linear(3, 512)
            elif _ +1 == 7:
                lin = nn.Linear(512, 1)
            else:
                lin = nn.Linear(512, 512)
    
            if _ == self.n_blocks - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(hidden_size), std=0.00001)
                torch.nn.init.constant_(lin.bias, -radius_init)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                            self.blocks.append(lin)'''
        #if(cfg.trainer == "trainers.w_normf_trainer"):
        self.blocks = nn.ModuleList()
        lin = nn.Linear(dim, hidden_size)
        torch.nn.init.constant_(lin.bias, 0.0)
        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden_size))
        self.blocks.append(lin)
    
        for _ in range(0, n_blocks):
            lin = nn.Linear(hidden_size, hidden_size)
            '''if (_ == self.n_blocks -2):
                print(lin.weight)
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(out_dim), std=0.00001)
                torch.nn.init.constant_(lin.bias, -radius_init)
            else:'''
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden_size))
            self.blocks.append(lin)
        lin = nn.Linear(hidden_size, out_dim)
        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(hidden_size), std=0.00001)
        torch.nn.init.constant_(lin.bias, -radius_init)
        self.blocks.append(lin)
        self.act = nn.ReLU()    
        
    #else:
        #for _ in range(n_blocks):
        #    self.blocks.append(nn.Linear(hidden_size, hidden_size))
        #self.blocks.append(nn.Linear(hidden_size, out_dim))
        #self.act = nn.ReLU()


        # Initialization
        #self.apply(lin_init)
        #self.blocks[0].apply(first_layer_lin_init)
        #if getattr(cfg, "zero_init_last_layer", False):
        #    print("true")
        #    torch.nn.init.constant_(self.blocks[-1].bias, 0.0)      

    def forward(self, x):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        net = x  # (bs, n_points, dim)
        for block in self.blocks[:-1]:
            net = self.act(block(net))
        out = self.blocks[-1](net)
        return out
 