import os
import torch
import importlib
import os.path as osp
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
#additions
import csv
import numpy as np
from torch import optim
from utils import load_imf
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
        points = torch.tensor(points)
        points = points.cuda()
        points.requires_grad = True
        dim = cfg.models.decoder.dim
        scaling = cfg.input.parameters.eta
        bs = cfg.input.parameters.bs
        cut_off_param = cfg.input.parameters.near_cut_off
        far_cut_off = cfg.input.parameters.far_cut_off
        inner_radius = cfg.input.parameters.inner_radius
        if (dim == 2):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
            x = np.random.uniform(-1.5,1.5,bs)
            y = np.random.uniform(-1.5,1.5,bs)
            xyz_list_1 = [None]*bs
            normed = [None]*bs
            norm_vector = [None]*bs
            phi_bound = np.random.uniform(0, 2*np.pi, 100)
            inner = [None]*100
            outer = [None]*100
            i = 0
            j = 0
            while j < 100:
                inner[j] = [0.1*np.cos(phi_bound[j]), 0.1*np.sin(phi_bound[j])]
                outer[j] = [1.5*np.cos(phi_bound[j]), 1.5*np.sin(phi_bound[j])]
                j = j+1
            while i < bs:
                xyz_list_1[i] = [x[i],y[i]]
                norm = np.linalg.norm(xyz_list_1[i], 2)
                norm_vector[i] = norm
                normed[i] = [x[i]/norm,y[i]/norm]
                
                i = i+1
            normed = np.float32(normed)
            xyz_list_1 = np.float32(xyz_list_1)
            xyz_1 = torch.tensor(xyz_list_1)
            xyz_1 = xyz_1.cuda()
            xyz_1.requires_grad = True
            
            inner = np.float32(inner)
            inner = torch.tensor(inner)
            inner = inner.cuda()
            inner.requires_grad = True
            
            outer = np.float32(outer)
            outer = torch.tensor(outer)
            outer = outer.cuda()
            outer.requires_grad = True
            
            grad_input_1 = gradient(input_net(xyz_1), xyz_1).view(bs, 2)
            norm_grad_input_1 = grad_input_1.norm(dim=-1)
            grad_input_1 = torch.transpose(grad_input_1, 0, 1)
            input_sample_1 = grad_input_1/norm_grad_input_1
            input_sample_1 = torch.transpose(input_sample_1, 0, 1)
            
            out_surf = self.net(points)      
            sdf_loss_on_surface = torch.square(out_surf).mean()
            grad_1 = gradient(self.net(xyz_1), xyz_1).view(bs, 2)
            p_1 = self.net(xyz_1)

            weight_1 = torch.transpose(eta(p_1), 0,1)
            weight_2 = torch.transpose(1 - eta(p_1), 0,1)
            
            norm_on_surf_loss_1 = (grad_1 - input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_2 = (grad_1 + input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())

            outer_loss = eta(self.net(outer)-0.1).mean()
            inner_loss = eta((-1)*self.net(inner)+0.1).mean()
            
            loss = sdf_loss_on_surface + field_loss + outer_loss + inner_loss
            if not no_update:
                loss.backward()
                self.opt.step()
        if(dim == 3):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
            #defining pointsample of domain and boundaries
            x = np.random.uniform(-1.3,1.3,bs)
            y = np.random.uniform(-1.3,1.3,bs)
            z = np.random.uniform(-1.3,1.3,bs)
           
            phi = np.random.uniform(0, 2*np.pi, 500)
            theta = np.random.uniform(0, np.pi, 500)
            xyz_list_1 = [None]*bs
            normed = [None]*bs
            norm_vector = [None]*bs
            outer = [None]*(500)
            inner = [None]*(500)
            i = 0
            j = 0
            domain_bound = 1.3
            inner_bound = np.linspace(0, inner_radius, 500)
            while j < 500:
                inner[j] = [inner_bound[j]*np.sin(theta[j])*np.cos(phi[j]), inner_bound[j]*np.sin(theta[j])*np.sin(phi[j]), inner_bound[j]*np.cos(theta[j])]
                outer[j] = [domain_bound*np.sin(theta[j])*np.cos(phi[j]), domain_bound*np.sin(theta[j])*np.sin(phi[j]), domain_bound*np.cos(theta[j])]
                j = j + 1
            inner = np.float32(inner)
            inner = torch.tensor(inner)
            inner = inner.cuda()
            inner.requires_grad = True
            
            outer = np.float32(outer)
            outer = torch.tensor(outer)
            outer = outer.cuda()
            outer.requires_grad = True
            
            while i < bs:
                xyz_list_1[i] = [x[i],y[i],z[i]]
                norm_vector[i] = np.linalg.norm(xyz_list_1[i], 2)
                normed[i] = [x[i]/norm_vector[i],y[i]/norm_vector[i],z[i]/norm_vector[i]]
                i = i+1
            xyz_list_1 = np.float32(xyz_list_1)
            xyz_1 = torch.tensor(xyz_list_1)
            xyz_1 = xyz_1.cuda()
            xyz_1.requires_grad = True
        
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
            #for sphere: 5
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
            #for sphere: < 0.5
            #choosing between nearby- and farfield
            norm_on_surf_loss_1 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1), norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.where(far_pot_cut < far_cut_off, torch.zeros_like(norm_on_surf_loss_1),norm_on_surf_loss_2)
            norm_on_surf_loss_1 = torch.square(norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.square(norm_on_surf_loss_2)         
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
             
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())
            
            #fixing the signs at the "boundaries", i.e. inner and outer sphere
            # also testing 10*scaling here
            #restult: it works better with the *10
            #new results show best results without add. scaling
            outer_loss = ((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean() #torch.abs(self.net(outer) + 0.5*torch.ones_like(self.net(inner))).mean()#((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean()
            inner_loss = ((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean() #torch.abs(self.net(inner) - 5*torch.ones_like(self.net(inner))).mean()#((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean()
            
            loss = 1000*sdf_loss_on_surface + field_loss + outer_loss + inner_loss #+ (1/100) *exp_plateau #+ 4*np.pi*0.1**2*inner_loss + 4*np.pi*1.1**2*outer_loss #10*plateau_loss.mean()  ##+ 0.1*torch.exp(-1e2 * torch.abs(out_surf)).mean()# 3e3, 1e2 
            
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
        #visualize = False
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

    def validate(self, input_net, far_net, points, cfg, *args, **kwargs):
        points = np.asarray(points)
        points = torch.tensor(points)
        points = points.cuda()
        points.requires_grad = True    
        point_sample = points 
        points = point_sample.tolist()
        input_net, cfg2 = load_imf(cfg.input.net_path, return_cfg=True)
        points = torch.tensor(points, device='cuda')
        points.requires_grad = True
        scaling = cfg.input.parameters.eta
        bs = 10000
        
        if(cfg.models.decoder.dim == 2):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
            x = np.random.uniform(-1.5,1.5,bs)#sample_r[i]*np.sin(sample_phi[i])
            y = np.random.uniform(-1.5,1.5,bs)
            phi_bound = np.random.uniform(0,2*np.pi,bs)
            xyz_list_1 = [None]*bs
            normed = [None]*bs
            norm_vector = [None]*bs
            outer = [None]*100
            inner = [None]*100
            i = 0
            while i < bs:
                xyz_list_1[i] = [x[i],y[i]]
                norm = np.linalg.norm(xyz_list_1[i], 2)
                norm_vector[i] = norm
                normed[i] = [x[i]/norm,y[i]/norm]
                i = i+1
            j = 0
            while j < 100:    
                inner[j] = [0.1*np.cos(phi_bound[j]), 0.1*np.sin(phi_bound[j])]
                outer[j] = [1.5*np.cos(phi_bound[j]), 1.5*np.sin(phi_bound[j])]
                j = j+1
            
                    
            xyz_list_1 = np.float32(xyz_list_1)
            xyz_1 = torch.tensor(xyz_list_1)
            xyz_1 = xyz_1.cuda()
            xyz_1.requires_grad = True
            
            grad_input_1 = gradient(input_net(xyz_1), xyz_1).view(bs, 2)
            norm_grad_input_1 = grad_input_1.norm(dim=-1)
            grad_input_1 = torch.transpose(grad_input_1, 0, 1)
            input_sample_1 = grad_input_1/norm_grad_input_1
            input_sample_1 = torch.transpose(input_sample_1, 0, 1)
            #####
            if(cfg.input.parameters.groundtruth == 1):
                index = 0
                grad_input_1 = [None]*bs
                while index < bs:
                    if(norm_vector[index] > 1):
                        grad_input_1[index] = [(-1)*normed[index][0],(-1)*normed[index][1]]
                    if(norm_vector[index] <= 1): 
                        grad_input_1[index] = normed[index]
                    index = index + 1
                #print(grad_input_1)
                grad_input_1 = torch.tensor(grad_input_1)
                grad_input_1 = grad_input_1.cuda()
                grad_input_1.requires_grad = True
                input_sample_1 = grad_input_1
                
            
            ###
            out_surf = self.net(points)

            sdf_loss_on_surface = torch.square(out_surf).mean()
            grad_1 = gradient(self.net(xyz_1), xyz_1).view(bs, 2)
            p_1 = self.net(xyz_1)
            
            weight_1 = torch.transpose(eta(p_1), 0,1)
            weight_2 = torch.transpose(1 - eta(p_1), 0,1)
            
            inner = np.float32(inner)
            inner = torch.tensor(inner)
            inner = inner.cuda()
            inner.requires_grad = True
            
            outer = np.float32(outer)
            outer = torch.tensor(outer)
            outer = outer.cuda()
            outer.requires_grad = True
            outer_loss = eta(self.net(outer)-0.1).mean()
            inner_loss = eta((-1)*self.net(inner)+0.1).mean()

            norm_on_surf_loss_1 = (grad_1 - input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_2 = (grad_1 + input_sample_1).norm(p = 2, dim=-1)
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
            
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())
            loss = sdf_loss_on_surface + field_loss + outer_loss + inner_loss
        if (cfg.models.decoder.dim == 3):
            def eta(x):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)

            x = np.random.uniform(-1.3,1.3,bs)
            y = np.random.uniform(-1.3,1.3,bs)
            z = np.random.uniform(-1.3,1.3,bs)
            
            phi = np.random.uniform(0, 2*np.pi, 500)
            theta = np.random.uniform(0, np.pi, 500)
            xyz_list_1 = [None]*bs
            normed = [None]*bs
            norm_vector = [None]*bs
            inner = [None]*(500)
            outer = [None]*(500)
            i = 0
            j = 0
            while j < 500:
                inner[j] = [0.01*np.sin(theta[j])*np.cos(phi[j]), 0.01*np.sin(theta[j])*np.sin(phi[j]), 0.01*np.cos(theta[j])]
                outer[j] = [1.1*np.sin(theta[j])*np.cos(phi[j]), 1.1*np.sin(theta[j])*np.sin(phi[j]), 1.1*np.cos(theta[j])]
                j = j + 1
            inner = np.float32(inner)
            inner = torch.tensor(inner)
            inner = inner.cuda()
            inner.requires_grad = True
            
            outer = np.float32(outer)
            outer = torch.tensor(outer)
            outer = outer.cuda()
            outer.requires_grad = True
            
            while i < bs:
                xyz_list_1[i] = [x[i],y[i],z[i]]
                norm_vector[i] = np.linalg.norm(xyz_list_1[i], 2)
                normed[i] = [x[i]/norm_vector[i],y[i]/norm_vector[i],z[i]/norm_vector[i]]
                i = i+1
            
            xyz_list_1 = np.float32(xyz_list_1)
            xyz_1 = torch.tensor(xyz_list_1)
            xyz_1 = xyz_1.cuda()
            xyz_1.requires_grad = True

            grad_input_1 = gradient(input_net(xyz_1), xyz_1).view(bs, 3)
            norm_grad_input_1 = grad_input_1.norm(dim=-1)
            grad_input_1 = torch.transpose(grad_input_1, 0, 1)
            input_sample_1 = grad_input_1/norm_grad_input_1
            input_sample_1 = torch.transpose(input_sample_1, 0, 1)
            
            pot_cut = input_net(xyz_1)
            
            out_surf = self.net(points)
            sdf_loss_on_surface = torch.square(out_surf).mean()
            grad_1 = gradient(self.net(xyz_1), xyz_1).view(bs, 3)
            p_1 = self.net(xyz_1)
            
            weight_1 = torch.transpose(eta(p_1), 0,1)
            weight_2 = torch.transpose(1 - eta(p_1), 0,1)
            pot_cut = input_net(xyz_1)
            far_pot_cut = far_net(xyz_1)
            #for sphere: 5
            cut_off_param = 15
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
            #for sphere: < 0.5
            #choosing between nearby- and farfield
            norm_on_surf_loss_1 = torch.where(far_pot_cut < 0.5, torch.zeros_like(norm_on_surf_loss_1), norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.where(far_pot_cut < 0.5, torch.zeros_like(norm_on_surf_loss_1),norm_on_surf_loss_2)
            norm_on_surf_loss_1 = torch.square(norm_on_surf_loss_1)
            norm_on_surf_loss_2 = torch.square(norm_on_surf_loss_2)         
            norm_on_surf_loss_1 = weight_1 * norm_on_surf_loss_1
            norm_on_surf_loss_2 = weight_2 * norm_on_surf_loss_2
             
            field_loss = (norm_on_surf_loss_1.mean()+norm_on_surf_loss_2.mean())

            outer_loss = ((torch.arctan((-1)*10*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean()
            inner_loss = ((torch.arctan(10*scaling*self.net(inner)+0.1) + np.pi/2)/np.pi).mean()
            
            loss = 1000* sdf_loss_on_surface + field_loss + outer_loss + inner_loss 
        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/sdf': sdf_loss_on_surface.detach().cpu().item(),
            'scalar/field': field_loss.detach().cpu().item()
        }
        #return {}

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
        save_name = "lowest_config_epoch_%s_iters_%s.pt" % (epoch, step)
        #torch.save(d, osp.join(self.cfg.save_dir, "checkpoints", save_name))
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
        ''''''#if self.sch is not None:
            #self.sch.step(epoch=epoch)
           #if writer is not None:
            #    writer.add_scalar(
            #        'train/opt_lr', self.sch.get_lr()[0], epoch)
