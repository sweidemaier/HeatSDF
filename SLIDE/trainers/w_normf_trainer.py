import os
import torch
import importlib
import os.path as osp
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
import numpy as np
import csv
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

    def update(self, points, input_net, far_net, cfg, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        points = points.clone().cuda().requires_grad_(True)
        
        dim = cfg.models.decoder.dim
        scaling = cfg.input.parameters.eta
        bs = cfg.input.parameters.bs
        cut_off_param = cfg.input.parameters.near_cut_off
        far_cut_off = cfg.input.parameters.far_cut_off
        inner_radius = cfg.input.parameters.inner_radius
        domain_bound = 1.3
        

        if (dim == 2):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
            xy = np.random.uniform(-1.3,1.3,(bs,2))
            phi_bound = np.random.uniform(0, 2*np.pi, 100)
            
            inner = np.column_stack((inner_radius*np.cos(phi_bound), inner_radius*np.sin(phi_bound)))
            outer = np.column_stack((domain_bound*np.cos(phi), domain_bound*np.sin(phi)))
            
            xyz_1 = torch.tensor(np.float32(xy)).cuda().requires_grad_(True)
            
            inner = torch.tensor(np.float32(inner)).cuda().requires_grad_(True)
            
            outer = torch.tensor(np.float32(outer)).cuda().requires_grad_(True)
            
            #sdf zero level set loss
            out_surf = self.net(points)      
            sdf_loss_on_surface = torch.square(out_surf).mean()
            
            #gradient loss
            grad_input_1 = gradient(input_net(xyz_1), xyz_1).view(bs, 2)
            norm_grad_input_1 = grad_input_1.norm(dim=-1)
            grad_input_1 = torch.transpose(grad_input_1, 0, 1)
            input_sample_1 = grad_input_1/norm_grad_input_1
            input_sample_1 = torch.transpose(input_sample_1, 0, 1) 
            grad_1 = gradient(self.net(xyz_1), xyz_1).view(bs, 2)
            p_1 = self.net(xyz_1)
            weight_1 = torch.transpose(eta(p_1), 0,1)
            weight_2 = torch.transpose(1 - eta(p_1), 0,1)
            pot_cut = input_net(xyz_1)
            far_pot_cut = far_net(xyz_1)
            norm_on_surf_loss_1 = (grad_1 - input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_2 = (grad_1 + input_sample_1).norm(p = 2, dim=-1)
            far_grad_input = gradient(far_net(xyz_1), xyz_1).view(bs, 3)
            far_norm_grad_input = far_grad_input.norm(dim=-1)
            far_grad_input = torch.transpose(far_grad_input, 0, 1)
            far_input_sample = far_grad_input/far_norm_grad_input
            far_input_sample = torch.transpose(far_input_sample, 0, 1)
            far_norm_1 = (grad_1 - far_input_sample).norm(p = 2, dim=-1)
            far_norm_2 = (grad_1 + far_input_sample).norm(p = 2, dim=-1)
            norm_on_surf_loss_1 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_1,  far_norm_1)
            norm_on_surf_loss_2 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_2, far_norm_2)

            #choosing between nearby- and farfield
            norm_on_surf_loss_1 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1), norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1),norm_on_surf_loss_2)
            norm_on_surf_loss_1 = torch.square(norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.square(norm_on_surf_loss_2)         
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())

            #fix signs inside and outside the surface
            outer_loss = ((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean() 
            inner_loss = ((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean() 

            loss = cfg.trainer.sdf_loss_weight*sdf_loss_on_surface + cfg.trainer.grad_norm_weight*field_loss + cfg.trainer.boundary_weight*outer_loss + cfg.trainer.boundary_weight*inner_loss 
            
            if not no_update:
                loss.backward()
                self.opt.step()
        if(dim == 3):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
            #defining pointsample of domain and boundaries
            xyz = np.random.uniform(-1.3,1.3,(bs, 3))
            phi = np.random.uniform(0, 2*np.pi, 500)
            theta = np.random.uniform(0, np.pi, 500)
            
            inner = np.column_stack((inner_radius*np.sin(theta)*np.cos(phi), inner_radius*np.sin(theta)*np.sin(phi), inner_radius*np.cos(theta)))
            outer = np.column_stack((domain_bound*np.sin(theta)*np.cos(phi), domain_bound*np.sin(theta)*np.sin(phi), domain_bound*np.cos(theta)))
            
            inner = torch.tensor(np.float32(inner)).cuda().requires_grad_(True)
            
            outer = torch.tensor(np.float32(outer)).cuda().requires_grad_(True)
            
            xyz_1 = torch.tensor(np.float32(xyz)).cuda().requires_grad_(True)
            
        
            #sdf zero level set loss
            out_surf = self.net(points)
            sdf_loss_on_surface = torch.square(out_surf).mean()
            
            #gradient loss
            grad_input_1 = gradient(input_net(xyz_1), xyz_1).view(bs, 3)
            norm_grad_input_1 = grad_input_1.norm(dim=-1)
            grad_input_1 = torch.transpose(grad_input_1, 0, 1)
            input_sample_1 = grad_input_1/norm_grad_input_1
            input_sample_1 = torch.transpose(input_sample_1, 0, 1)
            grad_1 = gradient(self.net(xyz_1), xyz_1).view(bs, 3)
            p_1 = self.net(xyz_1)
            weight_1 = torch.transpose(eta(p_1), 0,1)
            weight_2 = torch.transpose(1 - eta(p_1), 0,1)
            pot_cut = input_net(xyz_1)
            far_pot_cut = far_net(xyz_1)
            norm_on_surf_loss_1 = (grad_1 - input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_2 = (grad_1 + input_sample_1).norm(p = 2, dim=-1)
            far_grad_input = gradient(far_net(xyz_1), xyz_1).view(bs, 3)
            far_norm_grad_input = far_grad_input.norm(dim=-1)
            far_grad_input = torch.transpose(far_grad_input, 0, 1)
            far_input_sample = far_grad_input/far_norm_grad_input
            far_input_sample = torch.transpose(far_input_sample, 0, 1)
            far_norm_1 = (grad_1 - far_input_sample).norm(p = 2, dim=-1)
            far_norm_2 = (grad_1 + far_input_sample).norm(p = 2, dim=-1)
            norm_on_surf_loss_1 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_1,  far_norm_1)
            norm_on_surf_loss_2 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_2, far_norm_2)

            #choosing between nearby- and farfield
            norm_on_surf_loss_1 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1), norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1),norm_on_surf_loss_2)
            norm_on_surf_loss_1 = torch.square(norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.square(norm_on_surf_loss_2)         
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())
            
            #fix signs inside and outside the surface
            outer_loss = ((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean() 
            inner_loss = ((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean() 
            
            loss = cfg.trainer.sdf_loss_weight*sdf_loss_on_surface + cfg.trainer.grad_norm_weight*field_loss + cfg.trainer.boundary_weight*outer_loss + cfg.trainer.boundary_weight*inner_loss 
            
            if not no_update:
                loss.backward()
                self.opt.step()

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/sdf_loss_on_surface': sdf_loss_on_surface.detach().cpu().item(),    
            'scalar/field_loss': field_loss.detach().cpu().item()        
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
        print("Current learning rate: ", self.opt.param_groups[0]["lr"])
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

    def validate(self, input_net, far_net, points, cfg, *args, **kwargs):
        points = np.asarray(points)
        points = torch.tensor(points)
        points = points.cuda()
        points.requires_grad = True    
        point_sample = points 
        points = point_sample.tolist()
        points = torch.tensor(points, device='cuda')
        points.requires_grad = True
        scaling = cfg.input.parameters.eta
        bs = cfg.input.parameters.bs
        cut_off_param = cfg.input.parameters.near_cut_off
        far_cut_off = cfg.input.parameters.far_cut_off
        inner_radius = cfg.input.parameters.inner_radius
        domain_bound = 1.3
        
        if(cfg.models.decoder.dim == 2):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
            xy = np.random.uniform(-1.3,1.3,(bs,2))
            phi_bound = np.random.uniform(0, 2*np.pi, 100)
            
            inner = np.column_stack((inner_radius*np.cos(phi_bound), inner_radius*np.sin(phi_bound)))
            outer = np.column_stack((domain_bound*np.cos(phi), domain_bound*np.sin(phi)))
            
            xyz_1 = torch.tensor(np.float32(xy)).cuda().requires_grad_(True)
            
            inner = torch.tensor(np.float32(inner)).cuda().requires_grad_(True)
            
            outer = torch.tensor(np.float32(outer)).cuda().requires_grad_(True)
            
            #sdf zero level set loss
            out_surf = self.net(points)      
            sdf_loss_on_surface = torch.square(out_surf).mean()
            
            #gradient loss
            grad_input_1 = gradient(input_net(xyz_1), xyz_1).view(bs, 2)
            norm_grad_input_1 = grad_input_1.norm(dim=-1)
            grad_input_1 = torch.transpose(grad_input_1, 0, 1)
            input_sample_1 = grad_input_1/norm_grad_input_1
            input_sample_1 = torch.transpose(input_sample_1, 0, 1) 
            grad_1 = gradient(self.net(xyz_1), xyz_1).view(bs, 2)
            p_1 = self.net(xyz_1)
            weight_1 = torch.transpose(eta(p_1), 0,1)
            weight_2 = torch.transpose(1 - eta(p_1), 0,1)
            pot_cut = input_net(xyz_1)
            far_pot_cut = far_net(xyz_1)
            norm_on_surf_loss_1 = (grad_1 - input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_2 = (grad_1 + input_sample_1).norm(p = 2, dim=-1)
            far_grad_input = gradient(far_net(xyz_1), xyz_1).view(bs, 3)
            far_norm_grad_input = far_grad_input.norm(dim=-1)
            far_grad_input = torch.transpose(far_grad_input, 0, 1)
            far_input_sample = far_grad_input/far_norm_grad_input
            far_input_sample = torch.transpose(far_input_sample, 0, 1)
            far_norm_1 = (grad_1 - far_input_sample).norm(p = 2, dim=-1)
            far_norm_2 = (grad_1 + far_input_sample).norm(p = 2, dim=-1)
            norm_on_surf_loss_1 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_1,  far_norm_1)
            norm_on_surf_loss_2 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_2, far_norm_2)

            #choosing between nearby- and farfield
            norm_on_surf_loss_1 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1), norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1),norm_on_surf_loss_2)
            norm_on_surf_loss_1 = torch.square(norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.square(norm_on_surf_loss_2)         
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())

            #fix signs inside and outside the surface
            outer_loss = ((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean() 
            inner_loss = ((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean() 

            loss = cfg.trainer.sdf_loss_weight*sdf_loss_on_surface + cfg.trainer.grad_norm_weight*field_loss + cfg.trainer.boundary_weight*outer_loss + cfg.trainer.boundary_weight*inner_loss 
            
        if (cfg.models.decoder.dim == 3):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)

            xyz = np.random.uniform(-1.3,1.3,(bs,3))
            
            phi = np.random.uniform(0, 2*np.pi, 500)
            theta = np.random.uniform(0, np.pi, 500)
            
            inner = np.column_stack((inner_radius*np.sin(theta)*np.cos(phi), inner_radius*np.sin(theta)*np.sin(phi), inner_radius*np.cos(theta)))
            outer = np.column_stack((domain_bound*np.sin(theta)*np.cos(phi), domain_bound*np.sin(theta)*np.sin(phi), domain_bound*np.cos(theta)))
            

            xyz_1 = torch.tensor(np.float32(xyz)).cuda().requires_grad_(True)
            
            inner = torch.tensor(np.float32(inner)).cuda().requires_grad_(True)
            
            outer = torch.tensor(np.float32(outer)).cuda().requires_grad_(True)

            #sdf zero level set loss
            out_surf = self.net(points)
            sdf_loss_on_surface = torch.square(out_surf).mean()
            
            #gradient loss
            grad_input_1 = gradient(input_net(xyz_1), xyz_1).view(bs, 3)
            norm_grad_input_1 = grad_input_1.norm(dim=-1)
            grad_input_1 = torch.transpose(grad_input_1, 0, 1)
            input_sample_1 = grad_input_1/norm_grad_input_1
            input_sample_1 = torch.transpose(input_sample_1, 0, 1)
            grad_1 = gradient(self.net(xyz_1), xyz_1).view(bs, 3)
            p_1 = self.net(xyz_1)
            weight_1 = torch.transpose(eta(p_1), 0,1)
            weight_2 = torch.transpose(1 - eta(p_1), 0,1)
            pot_cut = input_net(xyz_1)
            far_pot_cut = far_net(xyz_1)
            norm_on_surf_loss_1 = (grad_1 - input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_2 = (grad_1 + input_sample_1).norm(p = 2, dim=-1)
            far_grad_input = gradient(far_net(xyz_1), xyz_1).view(bs, 3)
            far_norm_grad_input = far_grad_input.norm(dim=-1)
            far_grad_input = torch.transpose(far_grad_input, 0, 1)
            far_input_sample = far_grad_input/far_norm_grad_input
            far_input_sample = torch.transpose(far_input_sample, 0, 1)
            far_norm_1 = (grad_1 - far_input_sample).norm(p = 2, dim=-1)
            far_norm_2 = (grad_1 + far_input_sample).norm(p = 2, dim=-1)
            norm_on_surf_loss_1 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_1,  far_norm_1)
            norm_on_surf_loss_2 = torch.where(pot_cut > cut_off_param, norm_on_surf_loss_2, far_norm_2)

            #choosing between nearby- and farfield
            norm_on_surf_loss_1 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1), norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1),norm_on_surf_loss_2)
            norm_on_surf_loss_1 = torch.square(norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.square(norm_on_surf_loss_2)         
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())
            
            #fix signs inside and outside the surface
            outer_loss = ((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean() 
            inner_loss = ((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean() 
            
            loss = cfg.trainer.sdf_loss_weight*sdf_loss_on_surface + cfg.trainer.grad_norm_weight*field_loss + cfg.trainer.boundary_weight*outer_loss + cfg.trainer.boundary_weight*inner_loss 
        print("Validation successfull - Val. Loss:", loss.item())
        print("Surface Loss:", sdf_loss_on_surface.item())  
        print("Gradient Loss:", field_loss.item())
        print("Outer", outer_loss.item())
        print("Boundary Loss:", inner_loss.item())
        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/sdf': sdf_loss_on_surface.detach().cpu().item(),
            'scalar/field': field_loss.detach().cpu().item()
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

    def epoch_end(self, epoch,  writer=None, **kwargs):
        ''''''
