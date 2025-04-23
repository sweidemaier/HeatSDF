import os
import torch
import os.path as osp
from trainers.utils.diff_ops import gradient
from trainers.utils.diff_ops import manifold_gradient, manifold_jacobian, lapl_beltrami
from trainers.utils.diff_ops import jacobian as jac
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.new_utils import tens, spherical
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np
from trainers.utils.vis_utils import imf2mesh
from utils import load_imf
import importlib
class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        lib = importlib.import_module(cfg.models.decoder.type)
        iniz_net,_ = load_imf("/home/weidemaier/PDE Net/NFGP/logs/NeuralSDFs_2025-Jan-27-11-37-08", return_cfg=False)
        self.net = iniz_net
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
        
    def update(self, cfg, writer, epoch, phi, f_net, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        
    #Modeling Neural Deformations on Neural Surfaces
    
        ### samling
        bs = cfg.data.train.batch_size
        #if points == None:
        xy = np.random.uniform(-2.1,2.1, (bs, 3))
        #else: xy = points
        xy = tens(xy)
        
        tau = 0.025# 0.01
        
        bw = 0.05
        sample_size = 10000
        ### work to be done here
        a = 0.6
        b = 1.1
        c = 0.9
        sample_l = tens(np.random.uniform(-a,a, (sample_size,1)))
        sample_w = tens(np.random.uniform(-b,b, (sample_size,1)))
        sample_h = tens(np.random.uniform(-c,c, (sample_size,1)))
        ###
        sample = torch.cat((sample_l, sample_w, sample_h), 1)
        
        dist = torch.abs(phi(sample))
        keep = dist <= bw
        indices = keep.view(sample_size).nonzero()
        xy = sample[indices.view(indices.shape[0])]
        true_bs = xy.shape[0]
        
        while (true_bs < bs):
            sample_l = tens(np.random.uniform(-a,a, (sample_size,1)))
            sample_w = tens(np.random.uniform(-b,b, (sample_size,1)))
            sample_h = tens(np.random.uniform(-c,c, (sample_size,1)))
            sample = torch.cat((sample_l, sample_w, sample_h), 1)
            
            dist = torch.abs(phi(sample))
            keep = dist <= bw
            indices = keep.view(sample_size).nonzero()
            sample_in_range = sample[indices.view(indices.shape[0])]
             
            xy= torch.cat((xy, sample_in_range), 0)
            true_bs = xy.shape[0]
        
        bs = true_bs 
        
        writer.add_scalar('train/batch_size', bs, epoch)
        
        xyz = xy
        u = self.net(xy)
        phi = phi(xy)
        
        phi_grad = gradient(phi, xyz) 
        phi_grad_norm = torch.norm(phi_grad, dim = -1).view(xyz.shape[0], 1)             
        if(cfg.input.parameters.normalize == "normalize"):
            normal = phi_grad/phi_grad_norm
        else: normal = phi_grad
        d1u = manifold_gradient(u.reshape(bs), xyz, normal.cuda())
        
        #cutoff
        h = bw
        def beta(x, kappa):
            x = (-x + 2*kappa)/kappa
            vec = torch.where(x <= torch.zeros(x.shape).cuda(),  torch.zeros(x.shape).cuda(), -2*x**3 + 3*x**2)
            vec = torch.where(x > torch.ones(x.shape).cuda(), torch.ones(x.shape).cuda(), vec)
            return vec

        if(cfg.input.parameters.normalize == "normalize"):
            loss = (beta(torch.abs(phi), h/2)*phi_grad_norm*(torch.square(u).reshape(bs,1) + torch.square(torch.norm(d1u, p = 2, dim=(-1))).reshape(bs,1)))
        else:
            loss = (beta(torch.abs(phi), h/2)*(torch.square(u).reshape(bs,1) + tau*torch.square(torch.norm(d1u, p = 2, dim=(-1))).reshape(bs,1)))
        loss = loss.mean()
    
       
        if not no_update:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=0.5)
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
        #print("Current learning rate: ", self.opt.param_groups[0]["lr"])
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

    def validate(self,  cfg, writer, epoch, func,f_net,pts, *args, **kwargs):#, f_net
        ### samling
        bs = cfg.data.train.batch_size
        #if points == None:
        xy = np.random.uniform(-2.1,2.1, (bs, 3))
        #else: xy = points
        xy = tens(xy)
        dim_out = cfg.models.decoder.out_dim
        
        tau = 0.025
        bw = 0.05
        sample_size = 10000
        ### work to be done here
        a = 0.6
        b = 1.1
        c = 0.9
        sample_l = tens(np.random.uniform(-a,a, (sample_size,1)))
        sample_w = tens(np.random.uniform(-b,b, (sample_size,1)))
        sample_h = tens(np.random.uniform(-c,c, (sample_size,1)))
        ###
        sample = torch.cat((sample_l, sample_w, sample_h), 1)
        
        dist = torch.abs(phi(sample))
        keep = dist <= bw
        indices = keep.view(sample_size).nonzero()
        xy = sample[indices.view(indices.shape[0])]
        true_bs = xy.shape[0]
        
        while (true_bs < bs):
            sample_l = tens(np.random.uniform(-a,a, (sample_size,1)))
            sample_w = tens(np.random.uniform(-b,b, (sample_size,1)))
            sample_h = tens(np.random.uniform(-c,c, (sample_size,1)))
            sample = torch.cat((sample_l, sample_w, sample_h), 1)
            
            dist = torch.abs(phi(sample))
            keep = dist <= bw
            indices = keep.view(sample_size).nonzero()
            sample_in_range = sample[indices.view(indices.shape[0])]
             
            xy= torch.cat((xy, sample_in_range), 0)
            true_bs = xy.shape[0]
        
        bs = true_bs 
        
        writer.add_scalar('train/batch_size', bs, epoch)
        
        xyz = xy
        u = self.net(xy)
        phi = phi(xy)
        
        phi_grad = gradient(phi, xyz) 
        phi_grad_norm = torch.norm(phi_grad, dim = -1).view(xyz.shape[0], 1)             
        if(cfg.input.parameters.normalize == "normalize"):
            normal = phi_grad/phi_grad_norm
        else: normal = phi_grad
        d1u = manifold_gradient(u.reshape(bs), xyz, normal.cuda())
        
        #cutoff
        h = bw
        def beta(x, kappa):
            x = (-x + 2*kappa)/kappa
            vec = torch.where(x <= torch.zeros(x.shape).cuda(),  torch.zeros(x.shape).cuda(), -2*x**3 + 3*x**2)
            vec = torch.where(x > torch.ones(x.shape).cuda(), torch.ones(x.shape).cuda(), vec)
            return vec

        if(cfg.input.parameters.normalize == "normalize"):
            loss = (beta(torch.abs(phi), h/2)*phi_grad_norm*(-torch.square(u).reshape(bs,1) + torch.square(torch.norm(d1u, p = 2, dim=(-1))).reshape(bs,1)))
        else:
            loss = (beta(torch.abs(phi), h/2)*(-torch.square(u).reshape(bs,1) + tau*torch.square(torch.norm(d1u, p = 2, dim=(-1))).reshape(bs,1)))
        loss = loss.mean()
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

    def  save_best_val(self, epoch=None, step=None,**kwargs):
        d = {
            'opt': self.opt.state_dict(),
            'net': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        torch.save(d, osp.join(self.cfg.save_dir,  "best.pt"))

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
