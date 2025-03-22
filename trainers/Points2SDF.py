import os
import torch
import os.path as osp
import importlib
from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.new_utils import tens
import numpy as np
from utils import load_imf


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
        
    def update(self,  cfg, input_points, near_net, far_net, epoch, step,gt_inner, gt_outer, n_inner, n_outer, *args, **kwargs):
        if (epoch == 0):
            if(step == 1):
                print("init")
                
                return {}
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        domain_bound = 1.2
        bs = cfg.input.parameters.bs
        dims = cfg.models.decoder.dim 
        input_points = tens(input_points)
        factor = cfg.input.parameters.factors
        gamma = 500
        def eta(x, delta= 0.0005):
            vec = (1/4)*(x/(delta) + 2*torch.torch.ones_like(x))*(x/(delta) - torch.ones_like(x))**2
            vec = torch.where(x <= -(delta)*torch.ones_like(x),  torch.ones_like(x), vec)
            vec = torch.where(x > (delta)*torch.ones_like(x), torch.zeros_like(x), vec)
            return vec.view(x.shape[0], 1)
        def beta(x, kappa):
            vec = torch.where(x <= torch.zeros_like(x),  torch.zeros_like(x), -2*x**3 + 3*x**2)
            vec = torch.where(x > torch.ones_like(x), torch.zeros_like(x), vec)
            return vec
        if (dims == 3):
            ### sample points
            xyz = tens(np.random.uniform(-domain_bound, domain_bound, (bs, 3)))
            ### function values
            u = self.net(xyz)
            u_zero = self.net(input_points)
            u_zero_squared = torch.square(u_zero)
            grad_u = gradient(u, xyz)
            grad_u_near = gradient(near_net(xyz), xyz)
            grad_u_far = gradient(far_net(xyz), xyz)
            
            ### sort normals
            weight = eta(u)
            grad_blend = eta(torch.norm(grad_u_near, dim = -1) - gamma, delta = 0.1).view(bs, 1)*grad_u_far + (1-eta(torch.norm(grad_u_near, dim = -1) - gamma, delta = 0.1).view(bs, 1))*grad_u_near
            n = grad_blend/torch.norm(grad_blend, dim = -1).view(bs, 1)
            normal_alignment = (weight * torch.norm(grad_u - n, dim = -1).view(bs, 1) + (1-weight) * torch.norm(grad_u + n, dim = -1).view(bs, 1))
            
            ### boundary values
            gt_inner = tens(gt_inner)
            gt_outer = tens(gt_outer)
            n_inner = tens(n_inner)
            n_outer = tens(n_outer)
            grad_inner = gradient(self.net(gt_inner), gt_inner)
            grad_outer = gradient(self.net(gt_outer), gt_outer)
            
            inner_loss = torch.norm(grad_inner + n_inner, dim = -1)
            outer_loss = torch.norm(grad_outer - n_outer, dim = -1)
            bd_loss =  outer_loss.mean() + inner_loss.mean()
            
            
        scale = np.float32(cfg.input.parameters.param1)
           
        loss = scale*factor[0]*(u_zero_squared.mean()) + factor[1]*normal_alignment.mean() + factor[2]* bd_loss 
        
        if not no_update:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=2.0)
            self.opt.step()

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/surface': u_zero_squared.mean().detach().cpu().item(),
            'scalar/normal_alignement': normal_alignment.mean().detach().cpu().item(),
            'scalar/loss': loss.detach().mean().cpu().item(),
            
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

    def validate(self, cfg, input_points, near_net, far_net, writer, epoch, gt_inner, gt_outer,n_inner, n_outer, *args, **kwargs):
        domain_bound = 1.2
        bs = cfg.input.parameters.bs
        dims = cfg.models.decoder.dim 
        input_points = tens(input_points)
        factor = cfg.input.parameters.factors
        gamma = 500
        def eta(x, delta= 0.005):
            vec = (1/4)*(x/(delta) + 2*torch.torch.ones_like(x))*(x/(delta) - torch.ones_like(x))**2
            vec = torch.where(x <= -(delta)*torch.ones_like(x),  torch.ones_like(x), vec)
            vec = torch.where(x > (delta)*torch.ones_like(x), torch.zeros_like(x), vec)
            return vec.view(x.shape[0], 1)
        
        if (dims == 3):
            ### sample points
            xyz = tens(np.random.uniform(-domain_bound, domain_bound, (bs, 3)))
            ### function values
            u = self.net(xyz)
            u_zero = self.net(input_points)
            u_zero_squared = torch.square(u_zero)
            grad_u = gradient(u, xyz)
            grad_u_near = gradient(near_net(xyz), xyz)
            grad_u_far = gradient(far_net(xyz), xyz)
            
            ### sort normals
            weight_1 = eta(u)
            weight_2 = 1-eta(u)
            grad_blend = eta(torch.norm(grad_u_near, dim = -1) - gamma, delta = 500).view(bs, 1)*grad_u_far + (1-eta(torch.norm(grad_u_near, dim = -1) - gamma, delta = 500).view(bs, 1))*grad_u_near
            n = grad_blend/torch.norm(grad_blend, dim = -1).view(bs, 1)
            normal_alignment = (weight_1 * torch.norm(grad_u - n, dim = -1).view(bs, 1) + weight_2 * torch.norm(grad_u + n, dim = -1).view(bs, 1))
            
            ### boundary values
            gt_inner = tens(gt_inner)
            gt_outer = tens(gt_outer)
            n_inner = tens(n_inner)
            n_outer = tens(n_outer)
            grad_inner = gradient(self.net(gt_inner), gt_inner)
            grad_outer = gradient(self.net(gt_outer), gt_outer)
            inner_loss = torch.norm(grad_inner - n_inner, dim = -1)
            outer_loss = torch.norm(grad_outer + n_outer, dim = -1)
            bd_loss =  outer_loss.mean() + inner_loss.mean()
            
            
            
            
        scale = np.float32(cfg.input.parameters.param1)
           
        loss = scale*factor[0]*(u_zero_squared.mean()) + factor[1]*normal_alignment.mean() + factor[2]* bd_loss 
        
        writer.add_scalar('train/val_loss', loss.detach().cpu().item(), epoch)
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
