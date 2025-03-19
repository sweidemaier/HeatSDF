import os
import torch
import importlib
import os.path as osp
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.diff_ops import multi_jac
from trainers.utils.diff_ops import jacobian as jac
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.new_utils import tens

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
        
        
        bs =1000
        bs_bd = 100
        
        #parameters
        h = 1
        E = 70
        ny = 0.3

        L = 20

        lamda_x = 1
        lamda_y = 1

        #sampling 
        sample_x = np.random.uniform(0, L/2, bs)
        sample_y = np.random.uniform(0, L/2, bs)
        j = 0
        xy = [None]*bs
        while j < bs:
            xy[j] = [sample_x[j], sample_y[j]]
            j += 1
        xy = tens(xy)
        x = tens(sample_x)
        y = tens(sample_y)
        ###main
        C = E*h/(2*(1-ny**2))
        #jacob = jac(self.net(xy), xy)[0]
        
        jacob = jac(self.net(xy), xy)
        
        
        t_1 = jacob[:,0,0].unsqueeze(dim = -1)
        t_2 = jacob[:,1,1].unsqueeze(dim = -1)
        t_3 = jacob[:,0,0].unsqueeze(dim = -1)*jacob[:,1,1].unsqueeze(dim = -1)
        t_4 = jacob[:,0,1].unsqueeze(dim = -1) + jacob[:,1,0].unsqueeze(dim = -1)

        double_integral = C*(torch.square(t_1) + torch.square(t_2) + (2*ny)*t_3 + ((1-ny)/2)*torch.square(t_4)) 

        ###normal
        Lh_y = [None]*bs
        N_xx = [None]*bs
        j = 0
        while j < bs:
            Lh_y[j] = [L/2, sample_y[j]]
            ###based on the input assumption
            j += 1
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        N_xx = torch.cos(y*torch.pi/L)*h
        Lh_y = tens(Lh_y)
        
        normal_integral = N_xx * self.net(Lh_y)[:,0]
        ###bd
        bd_sample = np.random.uniform(0, L/2, bs_bd)
        zero_x = [None]*bs_bd
        zero_y = [None]*bs_bd
        i = 0
        while i < bs_bd:
            zero_x[i] = [bd_sample[i], 0]
            zero_y[i] = [0, bd_sample[i]]
            i += 1
        
        zero_x = tens(zero_x)
        zero_y = tens(zero_y)
        boundary_constr = lamda_x* torch.abs(self.net(zero_x)[0][1]).mean() + lamda_y* torch.abs(self.net(zero_y)[0][0]).mean()
        loss = double_integral.mean() - normal_integral.mean() + boundary_constr
        
        
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
