import os
import torch
import importlib
import numpy as np
import os.path as osp

from trainers.utils.diff_ops import gradient
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed


class Trainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))
        ### the network
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



    def update(self,  cfg, weights, input_points):
        self.net.train()
        self.opt.zero_grad()
        ### load settings
        bs = cfg.input.parameters.bs 
        tau = np.float32(cfg.input.parameters.tau)
        domain_bound = cfg.input.parameters.domain_bound
        input_bs = input_points.shape[0]

        ### sample uniform points
        xyz = (torch.rand(bs, 3, device='cuda', requires_grad=True) * 2 * domain_bound) - domain_bound
        
        ### compute terms for domain integral 
        u = self.net(xyz)
        u_squared = torch.square(u)
        u_grad_norm = torch.square(torch.norm(gradient(u, xyz), dim=-1))
        
        ### compute terms for surface integral
        u_input = self.net(input_points)
        val = weights.view(input_bs, 1)*(u_input.view(input_bs, 1))
        prod = torch.sum(val)    

        loss = u_squared.mean() + tau*u_grad_norm.mean() - 2*prod.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=2.0)
        self.opt.step()

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
        }


    def log_train(self, train_info, writer=None,
                  step=None, epoch=None):
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


    def validate(self, cfg, weights, input_points, writer, epoch):
        ### load settings
        bs = cfg.input.parameters.bs
        tau = np.float32(cfg.input.parameters.tau)
        domain_bound = cfg.input.parameters.domain_bound
        input_bs = input_points.shape[0]
        
        ### sample uniform points
        xyz = (torch.rand(bs, 3, device='cuda', requires_grad=True) * 2 * domain_bound) - domain_bound
        
        ### compute terms for domain integral
        u = self.net(xyz)
        u_squared = torch.square(u)
        u_grad_norm = torch.square(torch.norm(gradient(u, xyz), dim=-1))
        
        ###compute terms for surface integral
        u_input = self.net(input_points)
        val = weights.view(input_bs, 1)*(u_input.view(input_bs, 1))
        prod = torch.sum(val)     

        loss = u_squared.mean() + tau*u_grad_norm.mean() - 2*prod.mean()
        writer.add_scalar('train/val_loss', loss.detach().cpu().item(), epoch)
        return {
            'loss': loss.detach().cpu().item(),}  


    def save(self, epoch=None, step=None, appendix=None):
        #save current network weights after each iteration
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



    def save_best_val(self, epoch=None, step=None):
        # save network weight with lowest validation loass
        d = {
            'opt': self.opt.state_dict(),
            'net': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        torch.save(d, osp.join(self.cfg.save_dir,  "best.pt"))



    def resume(self, path, strict=True):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt['net'], strict=strict)
        self.opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch']
        return start_epoch
