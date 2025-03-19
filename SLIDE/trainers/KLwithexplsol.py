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
        bs_bd = 1000
        
        #parameters
        L = cfg.models.decoder.edge_l
        force = cfg.models.decoder.force
        E = 200.
        h = 0.01
        ny =0.3

        D = E*(2*h)**3/(12*(1-ny**2))

        def q(xy):
            return torch.sin(torch.pi*xy[:,0]/L)*torch.sin(torch.pi*xy[:,1]/L)
        ### samling
        xy = np.random.uniform(0,L, (bs, 2))
        xy = tens(xy)

        u = self.net(xy)
        #F = torch.ones(bs).cuda() #-0.025*torch.ones(bs).cuda() #q(xy)#/D
    ####
        
        D2_phi = laplace(u, xy)#jac(gradient(u[:,0], xy), xy)  
        D2_u_sq = torch.square(D2_phi) #torch.square(torch.norm(D2_phi, "fro", dim = (1,2)))

        bd_sample = np.random.uniform(0, L, bs_bd)
        bd_sample2 = np.random.uniform(0, 0.1, bs_bd)

        zero_x = [None]*bs_bd
        zero_y = [None]*bs_bd
        one_x = [None]*bs_bd
        one_y = [None]*bs_bd
        i = 0
        while i < bs_bd:
            zero_x[i] = [bd_sample[i], 0]
            zero_y[i] = [0, bd_sample[i]]
            one_x[i] = [bd_sample[i], L]
            one_y[i] = [L, bd_sample[i]]
            i += 1
        
        zero_x = tens(zero_x)
        zero_y = tens(zero_y)
        one_x = tens(one_x)
        one_y = tens(one_y)
       
        boundary_constr = torch.abs(self.net(zero_x)) + torch.abs(self.net(one_x)) + torch.abs(self.net(zero_y)) + torch.abs(self.net(one_y))

        F = 4*torch.pi**4*q(xy) #/D

        force_term = F.reshape(bs,1)*u


        loss = 1/2*D2_u_sq.mean() - force_term.mean()#  + boundary_constr.mean()
       
        
        
       
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
        writer.add_scalar('train/learning_rate', self.opt.param_groups[0]["lr"], writer_step)
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
        D = 1
        #q(xy)#/D
    #### here adaptations
        
        D2_phi = laplace(u, xy)#jac(gradient(u[:,0], xy), xy)  
        D2_u_sq = torch.square(D2_phi) #torch.square(torch.norm(D2_phi, "fro", dim = (1,2)))

        bd_sample = np.random.uniform(0, L, bs_bd)
        bd_sample2 = np.random.uniform(0, 0.1, bs_bd)

        zero_x = [None]*bs_bd
        zero_y = [None]*bs_bd
        one_x = [None]*bs_bd
        one_y = [None]*bs_bd
        i = 0
        while i < bs_bd:
            zero_x[i] = [bd_sample[i], 0]
            zero_y[i] = [0, bd_sample[i]]
            one_x[i] = [bd_sample[i], L]
            one_y[i] = [L, bd_sample[i]]
            i += 1
        
        zero_x = tens(zero_x)
        zero_y = tens(zero_y)
        one_x = tens(one_x)
        one_y = tens(one_y)
       
        boundary_constr = torch.abs(self.net(zero_x)) + torch.abs(self.net(one_x)) + torch.abs(self.net(zero_y)) + torch.abs(self.net(one_y))

        F = 4*torch.pi**4*q(xy) #/D

        loss = (1/2*D2_u_sq - (F.reshape(bs,1)*u)).mean() #+ 100*isom_loss.mean()
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
