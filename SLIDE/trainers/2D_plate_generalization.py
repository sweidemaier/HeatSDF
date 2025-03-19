import os
import torch
import importlib
import os.path as osp
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.diff_ops import laplace
from trainers.utils.diff_ops import jacobian as jac
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.new_utils import tens
import time
from models.borrowed_PINN_model import DR
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np
import functorch


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        #lib = importlib.import_module(cfg.models.decoder.type)
        #self.net = lib.Net(cfg, cfg.models.decoder)
        
        #arch = cfg.models.decoder.arch
        #arch =[50,'gelu',50,'gelu',50,'gelu']
        self.net =  DR(cfg.models.decoder.dim,cfg.models.decoder.arch,cfg.models.decoder.out_dim, cfg.models.decoder.bc) #, cfg.models.decoder.bc) 
        self.net.cuda()
        print("Net:")
        print(self.net)
        
        # The optimizer
        self.opt, self.sch = get_opt(
            self.net.parameters(), self.cfg.trainer.opt)

        # Prepare save directory
        os.makedirs(osp.join(cfg.save_dir, "val"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        
    def update(self, cfg, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        
        #Kirchhof-Love bending plates. net: R^2 -> R where net(xy) is the deflection z
        
        bs =1000
        bs_bd = 100
        
        #parameters
        L = cfg.models.decoder.edge_l

        lamda_x = 1
        lamda_y = 1

        #D = E*(2*h)**3/(12*1-ny**2)
        def q(xy):
            return 4*torch.pi**4*torch.sin(torch.pi*xy[:,0]/L)*torch.sin(torch.pi*xy[:,1]/L)
        ### samling
        xy = np.random.uniform(0,L, (bs, 2))
        xy = tens(xy)

        u = self.net(xy)
        F = -5*torch.ones(bs).cuda() #torch.zeros(bs).cuda() # #q(xy)#/D
    ####
        D2_phi_1 = jac(gradient(u[:,0], xy), xy)
        D2_phi_2 = jac(gradient(u[:,1], xy), xy)
        D2_phi_3 = jac(gradient(u[:,2], xy), xy)
        
        
        D2_u_sq = torch.square(torch.norm(D2_phi_1, "fro", dim = (1,2)))+ torch.square(torch.norm(D2_phi_2, "fro", dim = (1,2))) + torch.square(torch.norm(D2_phi_3, "fro", dim = (1,2)))
       
        

        bd_sample = np.random.uniform(0, L, bs_bd)
        

        zero_x = [None]*bs_bd
        zero_y = [None]*bs_bd
        one_x = [None]*bs_bd
        one_y = [None]*bs_bd
        i = 0
        while i < bs_bd:
            zero_x[i] = [bd_sample[i], 0]
            zero_y[i] = [0, bd_sample[i]]
            one_x[i] = [bd_sample[i], L]
            #one_y[i] = [L, bd_sample[i]]
            i += 1
        
        zero_x = tens(zero_x)
        zero_y = tens(zero_y)
        one_x = tens(one_x)
        one_y = tens(one_y)
        #self_zero_x = self.net(zero_y)
        #self_L_x = self.net(zero_x)

  
        #boundary_constr = lamda_x* torch.abs(self.net(zero_x) - torch.cat((0.1*torch.ones_like(self_zero_x[:,0].reshape(bs, 1)),torch.zeros_like(self_zero_x[:,0].reshape(bs, 1)),torch.zeros_like(self_zero_x[:,0].reshape(bs, 1))), 1).cuda()).mean() + lamda_x* torch.abs(self_L_x - torch.cat((0.1*torch.ones_like(self_L_x[:,0].reshape(bs, 1)),torch.zeros_like(self_zero_x[:,0].reshape(bs, 1)),torch.zeros_like(self_zero_x[:,0].reshape(bs, 1))), 1).cuda()).mean()# + torch.abs(self.net(one_x)-0.9*torch.ones_like).mean()#+ + lamda_y* torch.abs(self.net(zero_y)).mean() + torch.abs(self.net(one_y)).mean() + torch.abs(self.net(one_x)).mean()
        def net_ft(x):
            return self.net(x, ft = True)
        
        #def fcall(params, x):
            return torch.func.functional_call(self.net, params, x)
        #params = dict(self.net.named_parameters())
        #print(params)
        #Du = functorch.vmap(functorch.jacrev(net_ft))(xy)
        #D = torch.autograd.functional.jacobian(self.net, xy, vectorize=True)
        #print(D.shape)
        #Du = torch.sum(D, 2)#torch.einsum("aiaj->aij", D)
        D1 = gradient(u[:,0], xy).reshape(bs, 2,1)
        D2 = gradient(u[:,1], xy).reshape(bs, 2,1)
        D3 = gradient(u[:,2], xy).reshape(bs, 2,1)
        Du = torch.cat((D1,D2,D3), 2).transpose(1,2)
        
        #print(torch.transpose(torch.tensor([0,1]).cuda().repeat(bs, 1,1), 0,1).squeeze())
        #print(torch.transpose(torch.tensor([1,0,0]).cuda().repeat(bs, 1,1), 0,1).squeeze()*u)
        #print( torch.zeros(bs, 3).cuda().shape)
        #print(u.reshape(bs, 3,1))
        #print(torch.cat((u.reshape(bs, 3,1), torch.zeros(bs, 3,1).cuda()), 2))
        #bc = self.net.bc
        #print((Du*xy[:,0].view(bs, 1,1))[0])
        #print(xy[:,1].view(bs, 1,1)[0])
        #print((Du*xy[:,0].view(bs, 1,1)*xy[:,1].view(bs, 1,1))[0])
        #print(torch.cat((u.reshape(bs, 3,1)*xy[:,1].view(bs, 1,1), (u.reshape(bs, 3,1)*xy[:,0].view(bs, 1,1))), 2))
        #if (bc == "ls"):
        #    Du = (Du*xy[:,0].view(bs, 1,1)) + torch.cat((u.reshape(bs, 3,1), torch.zeros(bs, 3,1).cuda()), 2)
        #elif(bc == "ls-rs"):
        #    Du = Du*xy[:,0].view(bs, 1,1)*xy[:,1].view(bs, 1,1) + torch.cat((u.reshape(bs, 3,1)*xy[:,1].view(bs, 1,1), (u.reshape(bs, 3,1)*xy[:,0].view(bs, 1,1))), 2)
        #else:
        #    print("Boundary conditions not implemented.")
        #torch.transpose(torch.tensor([1,0,0]).cuda().repeat(bs, 1,1), 0,1).squeeze()*u + xy[:,0]*Du[







        #Du = functorch.vmap(functorch.jacrev(fcall))(params, xy) #.unsqueeze(0)) #[0,:,:,0,:]
        #print(torch.matmul(torch.eye(2,3).cuda().repeat(bs, 1, 1), Du).shape)

        DuId = torch.matmul(torch.eye(2,3).cuda().repeat(bs, 1, 1), Du)
        
        val = torch.matmul(torch.transpose(Du, 1,2), Du) + DuId + torch.transpose(DuId, 1,2)
        #print(val[0])
        #DuDu = assemble_gradDuDu(gradu_0 , gradu_1 , gradu_2 ,bs) #torch.matmul(torch.transpose(Du, 1, 2),Du) #[:,:,0,:]
        #print(DuDu[0])
        isom_loss = torch.square(torch.norm((val), "fro", dim = (1,2) ))
        eta()
        loss = (eta*(1/2*D2_u_sq - (F*u[:,2]) )).mean() + 10000*isom_loss.mean() #+ 10000*boundary_constr.mean()
       
        
        
       
        '''
        #compute strain
        jacob = jac(self.net(xyz), xyz)
        eps = [[jacob[:, 0,0], 0.5*(jacob[:, 0,1]+jacob[:, 1,0]), 0.5*(jacob[:, 0,1] +jacob[:, 2,0])],[0.5*(jacob[:, 0,1] + jacob[:,1,0]), jacob[:, 1,1], 0.5*(jacob[:, 1,2] +jacob[:, 2,1])],[0.5*(jacob[:, 2,0] + jacob[:,0,2]), 0.5*(jacob[:, 2,1]+jacob[:, 1,2]), jacob[:, 2,2]]]
        #compute stress
        k = 0
        i = 0
        j = 0
        delta = [None]*bs
        while k < bs:
            matrix = 
            while i < 
            delta[k] = matrix
            k += 1

        force_term = force*u
        lhs = torch.sum(eps*delta, 1) #maybe wrong
        lhs = torch.sum(lhs, 1)
        loss = h*lhs.mean() - force_term.mean() 
        '''
        if not no_update:
            loss.backward()
            self.opt.step()

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
        }

    def log_train(self, train_info, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return

        # Log training information to tensorboard
        writer_step = step if step is not None else epoch
        assert writer_step is not None
        for k, v in train_info.items():
            t, kn = k.split("/")[0], "/".join(k.split("/")[1:])
            if t not in ['scalar']:
                continue
            if t == 'scalar':
                writer.add_scalar('train/' + kn, v, writer_step)

        if visualize:
            with torch.no_grad():
                print("Visualize: %s" % step)
                res = int(getattr(self.cfg.trainer, "vis_mc_res", 256))
                thr = float(getattr(self.cfg.trainer, "vis_mc_thr", 0.))

                mesh = imf2mesh(
                    lambda x: self.net(x), res=res, threshold=thr)
                if mesh is not None:
                    save_name = "mesh_%diters.obj" \
                                % (step if step is not None else epoch)
                    mesh.export(osp.join(self.cfg.save_dir, "val", save_name))
                    mesh.export(osp.join(self.cfg.save_dir, "latest_mesh.obj"))

    def validate(self, *args, **kwargs):
         #Kirchhof-Love bending plates. net: R^2 -> R where net(xy) is the deflection z
        
        bs =10000
        bs_bd = 100
        
        #parameters
        h = 1
        E = 70
        ny = 0.3

        L = 1

        lamda_x = 1
        lamda_y = 1

        #D = E*(2*h)**3/(12*1-ny**2)
        def q(xy):
            return 4*torch.pi**4*torch.sin(torch.pi*xy[:,0]/L)*torch.sin(torch.pi*xy[:,1]/L)
        ### samling
        xy = np.random.uniform(0,L, (bs, 2))
        xy = tens(xy)

        u = self.net(xy)
        def q(xy):
            return 4*torch.pi**4*torch.sin(torch.pi*xy[:,0]/L)*torch.sin(torch.pi*xy[:,1]/L)
        ### samling
        xy = np.random.uniform(0,L, (bs, 2))
        xy = tens(xy)

        u = self.net(xy)
        F = -0.1*torch.ones(bs).cuda() #q(xy)#/D
    #### here adaptations
        D2_phi_1 = jac(gradient(u[:,0], xy), xy)
        D2_phi_2 = jac(gradient(u[:,1], xy), xy)
        D2_phi_3 = jac(gradient(u[:,2], xy), xy)
        
        lapl_u_sq = torch.square(torch.norm(D2_phi_1, "fro", dim = (1,2)))+ torch.square(torch.norm(D2_phi_2, "fro", dim = (1,2))) + torch.square(torch.norm(D2_phi_3, "fro", dim = (1,2)))
        
        bd_sample = np.random.uniform(0, L, bs_bd)
        zero_x = [None]*bs_bd
        zero_y = [None]*bs_bd
        one_x = [None]*bs_bd
        one_y = [None]*bs_bd
        i = 0
        while i < bs_bd:
            zero_x[i] = [bd_sample[i], 0]
            #zero_y[i] = [0, bd_sample[i]]
            #one_x[i] = [bd_sample[i], 1]
            #one_y[i] = [1, bd_sample[i]]
            i += 1
        
        zero_x = tens(zero_x)
        zero_y = tens(zero_y)
        one_x = tens(one_x)
        one_y = tens(one_y)
        boundary_constr = lamda_x* torch.abs(self.net(zero_x)).mean()#+ + lamda_y* torch.abs(self.net(zero_y)).mean() + torch.abs(self.net(one_y)).mean() + torch.abs(self.net(one_x)).mean()

  
        #def net_ft(x):
        #    return self.net(x, ft = True)
        
        #def fcall(params, x):
         #   return torch.func.functional_call(self.net, params, x)
        #params = dict(self.net.named_parameters())
        #print(params)
        #Du = functorch.vmap(functorch.jacrev(net_ft))(xy)
        #print(torch.transpose(torch.tensor([0,1]).cuda().repeat(bs, 1,1), 0,1).squeeze())
        #print(torch.transpose(torch.tensor([1,0,0]).cuda().repeat(bs, 1,1), 0,1).squeeze()*u)
        #print( torch.zeros(bs, 3).cuda().shape)
        #print(u.reshape(bs, 3,1))
        #print(torch.cat((u.reshape(bs, 3,1), torch.zeros(bs, 3,1).cuda()), 2))


       
        #bc = self.net.bc
        #print((Du*xy[:,0].view(bs, 1,1))[0])
        #print(xy[:,1].view(bs, 1,1)[0])
        #print((Du*xy[:,0].view(bs, 1,1)*xy[:,1].view(bs, 1,1))[0])
        #print(torch.cat((u.reshape(bs, 3,1)*xy[:,1].view(bs, 1,1), (u.reshape(bs, 3,1)*xy[:,0].view(bs, 1,1))), 2))
        #if (bc == "ls"):
        #    Du = (Du*xy[:,0].view(bs, 1,1)) + torch.cat((u.reshape(bs, 3,1), torch.zeros(bs, 3,1).cuda()), 2)
        #elif(bc == "ls-rs"):
        #    Du = Du*xy[:,0].view(bs, 1,1)*xy[:,1].view(bs, 1,1) + torch.cat((u.reshape(bs, 3,1)*xy[:,1].view(bs, 1,1), (u.reshape(bs, 3,1)*xy[:,0].view(bs, 1,1))), 2)
        #else:
        #    print("Boundary conditions not implemented.")
        #torch.transpose(torch.tensor([1,0,0]).cuda().repeat(bs, 1,1), 0,1).squeeze()*u + xy[:,0]*Du[


        #torch.transpose(torch.tensor([1,0,0]).cuda().repeat(bs, 1,1), 0,1).squeeze()*u + xy[:,0]*Du[







        #Du = functorch.vmap(functorch.jacrev(fcall))(params, xy) #.unsqueeze(0)) #[0,:,:,0,:]
        #print(torch.matmul(torch.eye(2,3).cuda().repeat(bs, 1, 1), Du).shape)

        #DuId = torch.matmul(torch.eye(2,3).cuda().repeat(bs, 1, 1), Du)
        #
        #val = torch.matmul(torch.transpose(Du, 1,2), Du) + DuId + torch.transpose(DuId, 1,2)
        #print(val[0])
        #DuDu = assemble_gradDuDu(gradu_0 , gradu_1 , gradu_2 ,bs) #torch.matmul(torch.transpose(Du, 1, 2),Du) #[:,:,0,:]
        #print(DuDu[0])
        #isom_loss = torch.square(torch.norm((val), "fro", dim = (1,2) ))
        #print(isom_loss.mean())
        #print(boundary_constr.mean())
        loss = (1/2*lapl_u_sq - (F*u[:,2]) ).mean() + 10000* boundary_constr.mean() #+ 100*isom_loss.mean()
        return {
            'loss': loss.detach().cpu().item(),}
        

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt': self.opt.state_dict(),
            'net': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        torch.save(d, osp.join(self.cfg.save_dir, "checkpoints", save_name))
        torch.save(d, osp.join(self.cfg.save_dir, "latest.pt"))

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt['net'], strict=strict)
        self.opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        '''if self.sch is not None:
            self.sch.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.sch.get_lr()[0], epoch)'''
