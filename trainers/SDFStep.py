import os
import torch
import sys
import trimesh
import numpy as np
import os.path as osp

from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.standard_utils import load_imf
from trainers.helper import sample_points_from_box_midpoints


class Trainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))
        ### we initialize the SDF-step with a precomputed approximation of the SDF of the unit sphere
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        init_config_path = os.path.join(base_dir, "configs", "initialization network")
        iniz_net,_ = load_imf(init_config_path)
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
        
    def update(self, cfg, input_points, near_net, far_net, epoch, step, gt_inner, gt_outer, kappa, box_points):
        if (epoch == 0):
            if(step == 1):
                print("init")
                return {}
            
        self.net.train()
        self.opt.zero_grad() 

        ### load settings        
        domain_bound = cfg.input.parameters.domain_bound
        bs = cfg.input.parameters.bs
        lamda = cfg.input.parameters.lamda
        box_width = 2*domain_bound/(cfg.input.parameters.box_count - 1)

        ### define cut-off functions
        def eta(x, delta= np.float32(cfg.input.parameters.delta)):
            vec = ((1/4)*(x/(delta) + 2*torch.ones(x.shape).cuda())*(x/(delta) - torch.ones(x.shape).cuda())**2)
            vec = torch.where(x <= -(delta)*torch.ones(x.shape).cuda(),  torch.ones(x.shape).cuda(), vec)
            vec = torch.where(x > (delta)*torch.ones(x.shape).cuda(), torch.zeros(x.shape).cuda(), vec)
            return vec.view(x.shape[0], 1)
        
        def beta(x, kappa):
            x = x/kappa
            vec = torch.where(x <= torch.zeros(x.shape).cuda(),  torch.zeros(x.shape).cuda(), -2*x**3 + 3*x**2)
            vec = torch.where(x > torch.ones(x.shape).cuda(), torch.ones(x.shape).cuda(), vec)
            return vec

        ### sample points
        if(cfg.input.parameters.sampling == "primitive"):
            xyz = (torch.rand(bs, 3, device='cuda', requires_grad=True) * 2 * domain_bound) - domain_bound
        elif(cfg.input.parameters.sampling == "boxes"):
            xyz = sample_points_from_box_midpoints(box_points, box_width, N = bs)
            xyz.requires_grad_(True)
        else: sys.exit("Sampling strategy not implemented!")
        inner_sample = sample_points_from_box_midpoints(gt_inner, box_width, N = 500)
        outer_sample = sample_points_from_box_midpoints(gt_outer, box_width, N = 500)

        ### comp surface loss, u and phi
        u = self.net(xyz)
        u_zero = self.net(input_points)
        u_zero_squared = torch.square(u_zero)
        grad_u = gradient(u, xyz)
        u_near = near_net(xyz)
        grad_u_near = gradient(u_near, xyz)

        ### sort normals
        weight = eta(u)
        if(far_net != None):
            u_far = far_net(xyz)
            grad_u_far = gradient(u_far, xyz)
            grad_blend = (1-beta(u_near, kappa))*grad_u_far + (beta(u_near, kappa))*grad_u_near
            n = grad_blend/torch.norm(grad_blend, dim = -1).view(bs, 1)
            normal_alignment = (weight * torch.square(torch.norm(grad_u + n, dim = -1)).view(bs, 1) + (1-weight) * torch.square(torch.norm(grad_u - n, dim = -1)).view(bs, 1))
        else:
            n = grad_u_near/torch.norm(grad_u_near, dim = -1).view(bs, 1)
            normal_alignment = (weight * torch.square(torch.norm(grad_u - n, dim = -1)).view(bs, 1) + (1-weight) * torch.square(torch.norm(grad_u + n, dim = -1)).view(bs, 1))
        
        ### boundary loss
        inner = self.net(inner_sample)
        outer = self.net(outer_sample)
        inner_loss = eta(inner, 0.0005) 
        outer_loss = eta(-outer, 0.0005)
        bd_loss =  outer_loss.mean() + inner_loss.mean()

        loss = lamda[0]*u_zero_squared.mean()  + lamda[1]*normal_alignment.mean()+ lamda[2]* bd_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=2.0)
        self.opt.step()

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/surface': u_zero_squared.mean().detach().cpu().item(),
            'scalar/normal_alignement': normal_alignment.mean().detach().cpu().item(),
            'scalar/loss': loss.detach().mean().cpu().item(),
            'scalar/bd_loss': bd_loss.detach().cpu().item(),
            
        }


    def log_train(self, train_info, writer=None,
                  step=None, epoch=None, visualize=False):
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

    def validate(self, cfg, input_points, near_net, far_net, writer, epoch, gt_inner, gt_outer, kappa, box_points):
        ###load settings
        domain_bound = cfg.input.parameters.domain_bound
        bs = 10*cfg.input.parameters.bs
        lamda = cfg.input.parameters.lamda
        box_width = 2*domain_bound/(cfg.input.parameters.box_count - 1)

        ### define cutoff function
        def eta(x, delta= np.float32(cfg.input.parameters.delta)):
            vec = ((1/4)*(x/(delta) + 2*torch.ones(x.shape).cuda())*(x/(delta) - torch.ones(x.shape).cuda())**2)
            vec = torch.where(x <= -(delta)*torch.ones(x.shape).cuda(),  torch.ones(x.shape).cuda(), vec)
            vec = torch.where(x > (delta)*torch.ones(x.shape).cuda(), torch.zeros(x.shape).cuda(), vec)

            return vec.view(x.shape[0], 1)
        def beta(x, kappa):
            x = x/kappa
            vec = torch.where(x <= torch.zeros(x.shape).cuda(),  torch.zeros(x.shape).cuda(), -2*x**3 + 3*x**2)
            vec = torch.where(x > torch.ones(x.shape).cuda(), torch.ones(x.shape).cuda(), vec)
            return vec

        ### sample points
        if(cfg.input.parameters.sampling == "primitive"):
            xyz = (torch.rand(bs, 3, device='cuda', requires_grad=True) * 2 * domain_bound) - domain_bound
        elif(cfg.input.parameters.sampling == "boxes"):
            xyz = sample_points_from_box_midpoints(box_points, box_width, N = bs)
            xyz.requires_grad_(True)
        else: sys.exit("Sampling strategy not implemented!")            
        inner_sample = sample_points_from_box_midpoints(gt_inner, box_width, N = 5000)
        outer_sample = sample_points_from_box_midpoints(gt_outer, box_width, N = 5000)

        ### comp surface loss, u and phi
        u = self.net(xyz)
        u_zero = self.net(input_points)
        u_zero_squared = torch.square(u_zero)
        grad_u = gradient(u, xyz)
        u_near = near_net(xyz)
        grad_u_near = gradient(u_near, xyz)

        ### sort normals
        weight = eta(u)
        if(far_net != None):
            u_far = far_net(xyz)
            grad_u_far = gradient(u_far, xyz)            
            grad_blend = (1-beta(u_near, kappa))*grad_u_far + (beta(u_near, kappa))*grad_u_near
            n = grad_blend/torch.norm(grad_blend, dim = -1).view(bs, 1)
            normal_alignment = (weight * torch.square(torch.norm(grad_u - n, dim = -1)).view(bs, 1) + (1-weight) * torch.square(torch.norm(grad_u + n, dim = -1)).view(bs, 1))
        else:
            n = grad_u_near/torch.norm(grad_u_near, dim = -1).view(bs, 1)
            normal_alignment = (weight * torch.square(torch.norm(grad_u - n, dim = -1)).view(bs, 1) + (1-weight) * torch.square(torch.norm(grad_u + n, dim = -1)).view(bs, 1))
        
        ### boundary loss
        inner = self.net(inner_sample)
        outer = self.net(outer_sample)
        inner_loss = eta(inner, 0.0005) 
        outer_loss = eta(-outer, 0.0005)
        bd_loss =  outer_loss.mean() + inner_loss.mean()
        
        loss = lamda[0]*u_zero_squared.mean()  + lamda[1]*normal_alignment.mean()+ lamda[2]* bd_loss
        
        
        writer.add_scalar('train/val_loss', loss.detach().cpu().item(), epoch)
        return {
            'loss': loss.detach().cpu().item(),}
        

    def save(self, epoch=None, step=None, appendix=None, vis = False):
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
        if vis:
            mesh = imf2mesh(self.net, res = 256, normalize=True, bound = 1.15, threshold=0.0)
            trimesh.exchange.export.export_mesh(mesh,self.cfg.save_dir +"/final_res" + ".obj", file_type=None, resolver=None) 
        torch.save(d, osp.join(self.cfg.save_dir, "checkpoints", save_name))
        torch.save(d, osp.join(self.cfg.save_dir, "latest.pt"))

    def  save_best_val(self, epoch=None, step=None):
        # save network weight with lowest validation loss
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
