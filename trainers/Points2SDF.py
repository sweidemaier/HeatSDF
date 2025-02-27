import os
import torch
import os.path as osp
import importlib
from trainers.utils.diff_ops import gradient, hessian
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.new_utils import tens
from models.borrowed_PINN_model import DR
from trainers import analyticSDFs
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np
from utils import load_imf
from trainers import analyticSDFs


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        lib = importlib.import_module(cfg.models.decoder.type)
        iniz_net,_ = load_imf("/home/weidemaier/PDE Net/NFGP/logs/NeuralSDFs_2025-Jan-27-11-37-08", return_cfg=False)
        self.net = iniz_net
        #self.net = lib.Net(cfg, cfg.models.decoder) ###
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
        
    def update(self,  cfg, input_points, near_net, far_net, epoch, step,*args, **kwargs):
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
        domain_bound = 1.5
        bs = cfg.input.parameters.bs
        dims = cfg.models.decoder.dim 
        input_points = tens(input_points)
        inner_radius = cfg.input.parameters.inner_radius
        outer_radius = 1.3#(domain_bound - 0.1)
        factor = cfg.input.parameters.factors
        cut_off_param = cfg.input.parameters.near_cut_off
        if (epoch < 50): #10
            #epoch_scaling = 50 #(1*(epoch+1))**2
        #elif (epoch < 9):
            epoch_scaling = 300*(epoch+1)/10
        else: epoch_scaling = 150
        epoch_scaling = np.float32(cfg.input.parameters.param2)
        cut_off_scaling = (1/0.3)*(epoch+1)
        def eta(x, scaling = epoch_scaling):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
        def eta2(x, cut_off = cut_off_scaling):
            delta = 1/cut_off
            vec = (-6)*(delta*x-0.5*torch.ones_like(x))**5 - 15*(delta*x-0.5*torch.ones_like(x))**4 - 10*(delta*x-0.5*torch.ones_like(x))**3
            vec = torch.where(x < -0.5*cut_off*torch.ones_like(x),torch.ones_like(x) , vec)
            vec = torch.where((x > 0.5*cut_off*torch.ones_like(x)),torch.zeros_like(x), vec )
            return vec
        def eta3(x):
            return torch.where(x < 0,torch.ones_like(x), torch.zeros_like(x))
            
            
            
        if (dims == 3):
            ### sample points
            xyz = tens(np.random.uniform(-domain_bound, domain_bound, (bs, 3)))
            phi = np.random.uniform(0, 2*np.pi, 500)
            theta = np.random.uniform(0, np.pi, 500)
            inner_large = tens(np.column_stack((0.25*np.sin(theta)*np.cos(phi), inner_radius*np.sin(theta)*np.sin(phi), inner_radius*np.cos(theta))))
            inner = tens(np.column_stack((inner_radius*np.sin(theta)*np.cos(phi), inner_radius*np.sin(theta)*np.sin(phi), inner_radius*np.cos(theta))))
            #translate = np.random.uniform(-0.25,0.5,500)
            #i = 0
            #vec = [None]*500
            #while i < 500:
            #    vec[i] = [0,translate[i], 0]
            #    i += 1
            #vec = tens(vec)
            #armadillo: [0.15,0.3,0.25]
            #acateon: [-0.2,-0.1,-0.15]
            #inner = inner +tens([-0.1, 0, -0.2]).reshape(1,3).repeat(500, 1)#+ vec.reshape(500,3)#tens([0,translate,0]).reshape(1,3).repeat(500, 1)
            #inner2 = inner + tens([-0.3,0.6,-0.2]).reshape(1,3).repeat(500, 1)
            #inner3 = inner + tens([-0.25,-0.7,0.3]).reshape(1,3).repeat(500, 1)
            #inner4 = inner + tens([0.8,0.5,-0.1]).reshape(1,3).repeat(500, 1)
            #inner5 = inner + tens([-0.5,0.5,-0.2]).reshape(1,3).repeat(500, 1)
            #inner = torch.cat((inner, inner2), 0)
            #tens([0., 0., 0.3])
            outer = tens(np.column_stack((outer_radius*np.sin(theta)*np.cos(phi), outer_radius*np.sin(theta)*np.sin(phi), outer_radius*np.cos(theta))))
            inner_radius_torus = 0.5
            inner_torus = tens(np.column_stack((inner_radius_torus*np.cos(phi), inner_radius_torus*np.sin(phi), np.zeros_like(np.sin(theta)))))
            #inner = inner_torus
            ### function values
            u = self.net(xyz)
            u_zero = self.net(input_points)
            u_zero_squared = torch.square(u_zero)
            grad_u = gradient(u, xyz)
            n_near = gradient(near_net(xyz), xyz)/torch.norm(gradient(near_net(xyz), xyz), dim = -1).view(bs, 1)
            n_far = gradient(far_net(xyz), xyz)/torch.norm(gradient(far_net(xyz), xyz), dim = -1).view(bs, 1)
            ### sort normals
            weight_1 = eta(u)#torch.transpose(eta(xyz), 0,1)
            weight_2 = 1-eta(u) #torch.transpose(1 - eta(xyz), 0,1)
            pot_cut = near_net(xyz)

            normal_alignment = (weight_1 * torch.norm(grad_u - n_near, dim = -1).view(bs, 1) + weight_2 * torch.norm(grad_u + n_near, dim = -1).view(bs, 1))
            #True
            #if (n_far != None):
            #    n_far = gradient(far_net(xyz), xyz)#/torch.norm(gradient(far_net(xyz), xyz), dim = -1).view(bs, 1)
            #    far_normal_alignment = (weight_1 * torch.norm(grad_u - n_far, dim = -1).view(bs, 1) + weight_2 * torch.norm(grad_u + n_far, dim = -1).view(bs, 1))
            #    normal_alignment = torch.where(pot_cut < cut_off_param*torch.ones(bs, 1).cuda(), normal_alignment, far_normal_alignment)
                #print(normal_alignment.shape)
            far_normal_alignment = (weight_1 * torch.norm(grad_u - n_far, dim = -1).view(bs, 1) + weight_2 * torch.norm(grad_u + n_far, dim = -1).view(bs, 1))
            
            ######
            
            normal_alignment = torch.where(pot_cut > cut_off_param*torch.ones(bs, 1).cuda(), normal_alignment, far_normal_alignment)

            #########
            ### fix SDF signes
            outer_loss = eta2(self.net(outer), 0.1) #((torch.arctan((-1)*(100)*self.net(outer)) + np.pi/2)/np.pi) #
            inner_loss = eta2(-self.net(inner), 0.01) #
            bd_loss =  outer_loss.mean() + inner_loss.mean()

            ### possibly introduce further loss terms
            eikonal = torch.abs(torch.norm(grad_u, dim = -1).view(bs,1) - torch.ones((bs,1)).cuda()) 
            
            xyz_hess = xyz.unsqueeze(0)
            #hess,_ = hessian(self.net(xyz_hess), xyz_hess)
            #hess = hess.reshape(bs,3,3)
            
            #singular_hess = torch.abs(torch.det(hess))
            #L2_hess = torch.norm(hess, p = "fro",dim = (1,2))
            
            if (epoch < 10):
                sing_fac = 1
            #elif (epoch < 20):
            #    sing_fac = 1 - (1)/(20-10)*(epoch - 10)
            else:
                sing_fac = 0
            if(epoch == 0):
                epoch_fac = 0
            else: epoch_fac = 1

            if (epoch < 3):
                scale = 1
            elif (epoch < 10):
                scale = 3*(epoch+1)/2
            elif (epoch < 50): scale = 100*(epoch+1)/5
            
            
        scale = np.float32(cfg.input.parameters.param1)
           
        loss = scale*factor[0]*(u_zero_squared.mean()) + factor[1]*normal_alignment.mean() + factor[2]* bd_loss #+ 0.001*sing_fac*singular_hess.mean()#+ 3*sing_fac*singular_hess.mean()
        ###loss = torch.abs(self.net(xyz) - analyticSDFs.phi_sphere_scaled(xyz)).mean()
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

    def validate(self, cfg, input_points, near_net, far_net, writer, epoch, *args, **kwargs):
        domain_bound = 1.5
        bs = cfg.input.parameters.bs
        dims = cfg.models.decoder.dim 
        input_points = tens(input_points)
        inner_radius = cfg.input.parameters.inner_radius
        outer_radius = 1.3#(domain_bound - 0.1)
        factor = cfg.input.parameters.factors
        cut_off_param = cfg.input.parameters.near_cut_off
        if (epoch < 50): #10
            #epoch_scaling = 50 #(1*(epoch+1))**2
        #elif (epoch < 9):
            epoch_scaling = 300*(epoch+1)/10
        else: epoch_scaling = 150
        epoch_scaling = np.float32(cfg.input.parameters.param2)
        cut_off_scaling = (1/0.3)*(epoch+1)
        def eta(x, scaling = epoch_scaling):
                return ((torch.arctan((-1)*scaling*x) + np.pi/2)/np.pi)
        def eta2(x, cut_off = cut_off_scaling):
            delta = 1/cut_off
            vec = (-6)*(delta*x-0.5*torch.ones_like(x))**5 - 15*(delta*x-0.5*torch.ones_like(x))**4 - 10*(delta*x-0.5*torch.ones_like(x))**3
            vec = torch.where(x < -0.5*cut_off*torch.ones_like(x),torch.ones_like(x) , vec)
            vec = torch.where((x > 0.5*cut_off*torch.ones_like(x)),torch.zeros_like(x), vec )
            return vec
        def eta3(x):
            return torch.where(x < 0,torch.ones_like(x), torch.zeros_like(x))
            
            
            
        if (dims == 3):
            ### sample points
            xyz = tens(np.random.uniform(-domain_bound, domain_bound, (bs, 3)))
            phi = np.random.uniform(0, 2*np.pi, 500)
            theta = np.random.uniform(0, np.pi, 500)
            inner_large = tens(np.column_stack((0.25*np.sin(theta)*np.cos(phi), inner_radius*np.sin(theta)*np.sin(phi), inner_radius*np.cos(theta))))
            inner = tens(np.column_stack((inner_radius*np.sin(theta)*np.cos(phi), inner_radius*np.sin(theta)*np.sin(phi), inner_radius*np.cos(theta))))
            #translate = np.random.uniform(-0.25,0.5,500)
            #i = 0
            #vec = [None]*500
            #while i < 500:
            #    vec[i] = [0,translate[i], 0]
            #    i += 1
            #vec = tens(vec)
            #armadillo: [0.15,0.3,0.25]
            #acateon: [-0.2,-0.1,-0.15]
            #noisy teddy: [-0.1, 0, -0.2]
            #inner = inner +tens([-0.1, 0, -0.2]).reshape(1,3).repeat(500, 1)#+ vec.reshape(500,3)#tens([0,translate,0]).reshape(1,3).repeat(500, 1)
            #inner2 = inner + tens([-0.3,0.6,-0.2]).reshape(1,3).repeat(500, 1)
            #inner3 = inner + tens([-0.25,-0.7,0.3]).reshape(1,3).repeat(500, 1)
            #inner4 = inner + tens([0.8,0.5,-0.1]).reshape(1,3).repeat(500, 1)
            #inner5 = inner + tens([-0.5,0.5,-0.2]).reshape(1,3).repeat(500, 1)
            #inner = torch.cat((inner, inner2), 0)
            #tens([0., 0., 0.3])
            outer = tens(np.column_stack((outer_radius*np.sin(theta)*np.cos(phi), outer_radius*np.sin(theta)*np.sin(phi), outer_radius*np.cos(theta))))
            inner_radius_torus = 0.5
            inner_torus = tens(np.column_stack((inner_radius_torus*np.cos(phi), inner_radius_torus*np.sin(phi), np.zeros_like(np.sin(theta)))))
            #inner = inner_torus
            ### function values
            u = self.net(xyz)
            u_zero = self.net(input_points)
            u_zero_squared = torch.square(u_zero)
            grad_u = gradient(u, xyz)
            n_near = gradient(near_net(xyz), xyz)/torch.norm(gradient(near_net(xyz), xyz), dim = -1).view(bs, 1)
            n_far = gradient(far_net(xyz), xyz)/torch.norm(gradient(far_net(xyz), xyz), dim = -1).view(bs, 1)
            ### sort normals
            weight_1 = eta(u)#torch.transpose(eta(xyz), 0,1)
            weight_2 = 1-eta(u) #torch.transpose(1 - eta(xyz), 0,1)
            pot_cut = near_net(xyz)

            normal_alignment = (weight_1 * torch.norm(grad_u - n_near, dim = -1).view(bs, 1) + weight_2 * torch.norm(grad_u + n_near, dim = -1).view(bs, 1))
            #True
            #if (n_far != None):
            #    n_far = gradient(far_net(xyz), xyz)#/torch.norm(gradient(far_net(xyz), xyz), dim = -1).view(bs, 1)
            #    far_normal_alignment = (weight_1 * torch.norm(grad_u - n_far, dim = -1).view(bs, 1) + weight_2 * torch.norm(grad_u + n_far, dim = -1).view(bs, 1))
            #    normal_alignment = torch.where(pot_cut < cut_off_param*torch.ones(bs, 1).cuda(), normal_alignment, far_normal_alignment)
                #print(normal_alignment.shape)
            far_normal_alignment = (weight_1 * torch.norm(grad_u - n_far, dim = -1).view(bs, 1) + weight_2 * torch.norm(grad_u + n_far, dim = -1).view(bs, 1))
            
            ######
            
            normal_alignment = torch.where(pot_cut > cut_off_param*torch.ones(bs, 1).cuda(), normal_alignment, far_normal_alignment)

            #########
            ### fix SDF signes
            outer_loss = eta2(self.net(outer), 0.1) #((torch.arctan((-1)*(100)*self.net(outer)) + np.pi/2)/np.pi) #
            inner_loss = eta2(-self.net(inner), 0.01) #
            bd_loss =  outer_loss.mean() + inner_loss.mean()

            ### possibly introduce further loss terms
            eikonal = torch.abs(torch.norm(grad_u, dim = -1).view(bs,1) - torch.ones((bs,1)).cuda()) 
            
            xyz_hess = xyz.unsqueeze(0)
            #hess,_ = hessian(self.net(xyz_hess), xyz_hess)
            #hess = hess.reshape(bs,3,3)
            
            #singular_hess = torch.abs(torch.det(hess))
            #L2_hess = torch.norm(hess, p = "fro",dim = (1,2))
            
            if (epoch < 10):
                sing_fac = 1
            #elif (epoch < 20):
            #    sing_fac = 1 - (1)/(20-10)*(epoch - 10)
            else:
                sing_fac = 0
            if(epoch == 0):
                epoch_fac = 0
            else: epoch_fac = 1

            if (epoch < 3):
                scale = 1
            elif (epoch < 10):
                scale = 3*(epoch+1)/2
            elif (epoch < 50): scale = 100*(epoch+1)/5
            
            
        scale = np.float32(cfg.input.parameters.param2)
            
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
