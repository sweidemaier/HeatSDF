import os
import torch
import importlib
import os.path as osp
import torch.nn.functional as F
import numpy as np
from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.helper import comp_weights
from torch import optim

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
    
    def update(self, pointcloud, weights, cfg, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        bs = cfg.input.parameters.bs
        dims = cfg.models.decoder.dim 

        if (dims == 2):
            bs = 1000
            xy = np.random.uniform(-1.3,1.3,(bs, 2))
            xyz_list = np.float32(xy)
            xyz = torch.tensor(xyz_list)
            xyz = xyz.cuda()
            xyz.requires_grad = True
            
            input_surf_points = pointcloud
            tau = np.float32(cfg.input.parameters.tau)
            u = self.net(xyz)
            u_squared = torch.square(u)
            
            grad_u_norm = gradient(self.net(xyz), xyz).view(
                    bs, -1, xyz.size(-1)).norm(p = 2, dim=-1)
            grad_u_squared = torch.square(grad_u_norm)
            
            tensor_input_p = torch.tensor(input_surf_points, device='cuda')
            eval_p = self.net(tensor_input_p)
            loss_1 = u_squared + tau*grad_u_squared 
            loss_1 = loss_1.mean()
            weight = torch.tensor(weights)
            weight = weight.unsqueeze(1)
            weight = weight.cuda()
            val = torch.mul(weight,eval_p)
            sprod = torch.sum(val)
            
            loss = loss_1 - 2*sprod 
        
        if (dims == 3):
            xyz = np.random.uniform(-1.3,1.3, (bs, 3))
            
            xyz_list = np.float32(xyz)
            xyz = torch.tensor(xyz_list)
            xyz = xyz.cuda()
            xyz.requires_grad = True
            
            input_surf_points = pointcloud
            tau = np.float32(cfg.input.parameters.tau)
            
            u = self.net(xyz)
            u_squared = torch.square(u)

            grad_u_norm = gradient(self.net(xyz), xyz).view(
                    bs, -1, xyz.size(-1)).norm(p = 2, dim=-1)
            grad_u_squared = torch.square(grad_u_norm)

            tensor_input_p = torch.tensor(input_surf_points, device='cuda')
            eval_p = self.net(tensor_input_p)
            loss = u_squared + tau*grad_u_squared 
            loss = loss.mean()
            weight = torch.tensor(weights)
            weight = weight.unsqueeze(1)
            weight = weight.cuda()
            val = torch.mul(weight,eval_p)
           
            sprod = torch.sum(val)
            
            loss = loss - 2*sprod   
                
        if not no_update:
            loss.backward()
            self.opt.step()
            
        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/sprod': sprod.detach().cpu().item(),
        }

    def log_train(self, train_info, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return
        visualize = False
        # Log training information to tensorboard
        writer_step = step if step is not None else epoch
        assert writer_step is not None
        for k, v in train_info.items():
            t, kn = k.split("/")[0], "/".join(k.split("/")[1:])
            if t not in ['scalar']:
                continue
            if t == 'scalar':
                writer.add_scalar('train/' + kn, v, writer_step)
        print(self.opt.param_groups[0]["lr"])
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

    def validate(self, weights, pointcloud, cfg, *args, **kwargs):
        if (cfg.models.decoder.dim == 2):
            bs = 10000
            xy = np.random.uniform(-1.3,1.3, (bs,2))
            
            xyz_list = np.float32(xy)
            xyz = torch.tensor(xyz_list)
            xyz = xyz.cuda()
            xyz.requires_grad = True
            input_surf_points = pointcloud
            tau = cfg.input.parameters.tau
            u = self.net(xyz)
            u_squared = torch.square(u)
            grad_u_norm = gradient(self.net(xyz), xyz).view(
                    bs, -1, xyz.size(-1)).norm(dim=-1)

            grad_u_squared = torch.square(grad_u_norm)
            tensor_input_p = torch.tensor(input_surf_points, device='cuda')
            tensor_input_p.requires_grad = True
            eval = self.net(tensor_input_p)
            loss_1 = u_squared + tau*grad_u_squared 
            loss_1 = loss_1.mean()
            weight = torch.tensor(weights)
            weight = weight.unsqueeze(1)
            weight = weight.cuda()
            val = torch.mul(weight,eval)
            sprod = torch.sum(val)
            loss = loss_1 - 2*sprod
        if (cfg.models.decoder.dim == 3):
            bs = 100000
            xyz = np.random.uniform(-1.3,1.3, (bs,3))
            xyz_list = np.float32(xyz)
            xyz = torch.tensor(xyz_list)
            xyz = xyz.cuda()
            xyz.requires_grad = True
            
            input_surf_points = pointcloud
            tau = np.float32(cfg.input.parameters.tau)
            
            u = self.net(xyz)
            u_squared = torch.square(u)

            grad_u_norm = gradient(self.net(xyz), xyz).view(
                    bs, -1, xyz.size(-1)).norm(p = 2, dim=-1)
            grad_u_squared = torch.square(grad_u_norm)

            tensor_input_p = torch.tensor(input_surf_points, device='cuda')
            eval_p = self.net(tensor_input_p)
            loss = u_squared + tau*grad_u_squared 
            loss = loss.mean()
            weight = torch.tensor(weights)
            weight = weight.unsqueeze(1)
            weight = weight.cuda()
            val = torch.mul(weight,eval_p)
            sprod = torch.sum(val)
            
            loss = loss - 2*sprod
        #print("Validation successfull - Val. Loss:", loss.item())
        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/sprod': sprod.detach().cpu().item()
        }        
    
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

    def save_lowest(self, epoch=None, step=None, **kwargs):
        d = {
            'opt': self.opt.state_dict(),
            'net': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        torch.save(d, osp.join(self.cfg.save_dir, "lowest.pt"))

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt['net'], strict=strict)
        self.opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.sch is None:
            self.sch.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.sch.get_lr()[0], epoch)
