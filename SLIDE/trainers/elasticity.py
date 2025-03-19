import os
import torch
import importlib
import os.path as osp
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.diff_ops import D2
from trainers.utils.diff_ops import D3
from trainers.utils.diff_ops import D4
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
import numpy as np

class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        lib = importlib.import_module(cfg.models.decoder.type)
        self.net = lib.Net(cfg, cfg.models.decoder)
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
        
    def update(self, data, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        def deriv(self, t, n=1):
            dx = self.net(t)
            for i in range(n):
                dx = torch.autograd.grad(outputs=dx, inputs=t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
            return dx

        bs = 100
        #parameters
        L = 1.
        q = 1. # 1 #1.
        alpha = 1.
        lambda_1 = 100.
        lambda_2 = 100.
        lambda_3 = 100.
        lambda_4 = 100.
        #sampling 
        sample = np.random.uniform(0, 1, bs)
        #sample = np.linspace(0,1,bs)
        
        x = [None]*bs
        i = 0
        while i in range(bs):
            x[i] = [sample[i]]
            i = i+1
        #evaluations
        x = np.float32(x)
        x = torch.tensor(x)
        x = x.cuda()
        
        x.requires_grad = True
        phi = self.net(x)
        print(phi[0])
        d2_phi = deriv(self, x, n = 2)
        print(d2_phi[0])
        d2_phi_sq = torch.square(d2_phi)
        
        
        end = [L]
        end = torch.tensor(end)
        end = end.cuda()
        end.requires_grad = True
        phi_2prim_L = deriv(self, end, n = 2)
        phi_3prim_L = deriv(self, end, n = 3)               
        begin = [0.]
        begin = torch.tensor(begin)
        begin = begin.cuda()
        begin.requires_grad = True
        phi_0 = self.net(begin)
        phi_L = self.net(end)
        phi_prim_0 = gradient(self.net(begin), begin).view(
                            1, -1, end.size(-1))
        phi_prim_L = gradient(self.net(end), end).view(
                            1, -1, end.size(-1))
        #loss_bd_1 = phi*torch.transpose(D3(self.net(begin), begin).view(1, -1, x.size(-1)), 0, 1)
        #loss_bd_2 = phi*torch.transpose(D2(self.net(begin), begin).view(1, -1, x.size(-1)), 0, 1)
        loss = alpha/2* d2_phi_sq.mean() + q*phi.mean() -  +lambda_1 * torch.abs(phi_0-torch.ones_like(phi_0)) + lambda_2*torch.abs(phi_prim_0) + lambda_3*phi_2prim_L**2 + lambda_4*phi_3prim_L**2 #lambda_3 * torch.abs(phi_L) + lambda_4 * torch.abs(phi_prim_L)#+ lambda_3*phi_2prim_L**2 + lambda_4*phi_3prim_L**2 #+ lambda_1 * phi_0**2 + lambda_1*phi_L**2 + phi_prim_0**2 + phi_prim_L**2 #  alpha*loss_bd_1.mean()- alpha*loss_bd_2.mean() + lambda_1 * phi_0**2 + lambda_2*phi_prim_0**2 + lambda_3*phi_2prim_L**2 + lambda_4*phi_3prim_L**2 
        
        # -  alpha*loss_bd_1.mean()+ alpha*loss_bd_2.mean()
        # print(phi.mean())

        #version where we directly plug in the pde
                
        #loss = l_eq  + l_bd
        if not no_update:
            loss.backward()
            self.opt.step()

        return {
            'loss': loss.detach().cpu().item(),
            
            'scalar/loss': loss.detach().cpu().item(),
        }

    def log_train(self, train_info, train_data, writer=None,
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

    def validate(self, test_loader, epoch, *args, **kwargs):
        return {}

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
        if self.sch is not None:
            self.sch.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.sch.get_lr()[0], epoch)
