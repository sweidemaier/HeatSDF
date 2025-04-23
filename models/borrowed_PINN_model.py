import torch
import torch.nn as nn
#from trainers import analyticSDFs
torch.pi = torch.acos(torch.zeros(1)).item() * 2


class DR(nn.Module):
    def __init__(self, inputDim, arch, outputDim, bc):
        super(DR,self).__init__()
        self.model = torch.nn.Sequential()
        currDim = inputDim
        layerCount = 1
        activCount = 1
        for i in range(len(arch)):
            if(type(arch[i]) == int):
                self.model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,arch[i],bias=True))
                currDim = arch[i]
                layerCount += 1
            elif(type(arch[i]) == str):
                self.model.add_module('activ '+str(activCount),nn.ReLU())
                activCount += 1
        self.model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,outputDim,bias=True))
        self.bc = bc

    def forward(self, x, bd = False):
        out = self.model(x)
        u = torch.zeros(x.shape[0], 3).cuda()
        #print(out)
        #print((x))
        if self.bc[0] == 'no':
            return out
        if self.bc[0] == 'ls':
            if self.bc[1] == 'clap':
                trial_func = x[:,0]*x[:,0]
            else:
                trial_func = x[:,0]
        elif self.bc[0] == 'rs':
            trial_func = x[:,1]
        elif self.bc[0] == 'test':
            trial_func = torch.sin(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])
            return out * trial_func[:, None] 
        elif self.bc[0] == 'ls-rs':
            trial_func = (x[:,0]) * (x[:,1])
        elif self.bc[0] == 'simpl_sup':

            trial_func = (x[:,0]) * (x[:,1]) * (x[:,0] - 1) * (x[:,1] - 1)
            #trial_func = (torch.abs(x[:,0]-0.5)-0.5)*(torch.abs(x[:,1]-0.5)-0.5) 
        elif self.bc[0] == 'fc':
            trial_func = (x[:,0]**2-0.25) * (x[:,1]**2-0.25)
        elif self.bc[0] == 'fc_circular':
            trial_func = (1.-x[:,0]*x[:,0]-x[:,1]*x[:,1])/2.
        elif self.bc[0] == 'displ':
            trial_func_1 = x[:,0] 
            g_1 = -(x[:,0] + 0.1*torch.ones_like(x[:,0]))*(x[:,0] - torch.ones_like(x[:,0])) + (x[:,0] - 1.1*torch.ones_like(x[:,0]))*x[:,0]
            g_1 = g_1.reshape(x.shape[0], 1)
            g = torch.cat((g_1, torch.zeros_like(g_1), torch.zeros_like(g_1)), 1)
            out = out * torch.square((trial_func_1[:, None])*(trial_func_1[:, None] - torch.ones_like(trial_func_1[:, None]))) + g
            '''out_0 = torch.where(x[:,0] < 0.1*torch.ones_like(x[:,0]), out[:,0] + 0.1*torch.ones_like(out[:,0]), out[:,0]).reshape(x.shape[0], 1)
            out_1 = torch.where(x[:,0] > 0.9*torch.ones_like(x[:,0]), out[:,0] - 0.1*torch.ones_like(out[:,0]), out[:,0]).reshape(x.shape[0], 1)

            out_shift = out_0 + out_1
            out_1 = torch.where(x[:,0] < 0.1*torch.ones_like(x[:,0]), torch.zeros_like(out[:,1]).cuda(), out[:,1])
            out_1 = torch.where(x[:,0] > 0.9*torch.ones_like(x[:,0]), torch.zeros_like(out_1).cuda(), out_1).reshape(x.shape[0], 1)
            
            out_2 = torch.where(x[:,0] < 0.1*torch.ones_like(x[:,0]), torch.zeros_like(out[:,1]).cuda(), out[:,2])
            out_2 = torch.where(x[:,0] > 0.9*torch.ones_like(x[:,0]), torch.zeros_like(out[:,1]).cuda(), out_2).reshape(x.shape[0], 1)
            
            out = torch.cat((out_shift, out_1, out_2), 1) 
            ''' 
            return out  
        elif self.bc[0] == 'implicit':
            x = x.squeeze()
            if self.bc[1] == "sphere":
                implicit_func = analyticSDFs.phi_func_sphere
            elif self.bc[1] == "comp_FEM":
                implicit_func = analyticSDFs.comp_FEM
            
            else: print("Error. Implicit Function not implemented")
            return (out*implicit_func(x).view(x.shape[0], 1)).reshape(x.shape[0])  
            
            #shift_0 = torch.where(x[:,0] < 0.1*torch.ones_like(x[:,0]), torch.tensor([0.1]).cuda(), torch.zeros(1, 1).cuda()).reshape(x.shape[0], 1)
            #shift_L = torch.where(x[:,0] > 0.9*torch.ones_like(x[:,0]), torch.tensor([-0.1]).cuda(), torch.zeros(1, 1).cuda()).reshape(x.shape[0], 1) 
            #shift = torch.cat((shift_0 + shift_L, torch.zeros(x.shape[0], 2).cuda()), 1)
            #return (out + shift)
            
        else:
            raise ValueError('Missing Dirichlet boundary conditions.')

        # broadcast over all five fields
        u = out * trial_func[:, None] 
        print("testing if we are here")
        return u