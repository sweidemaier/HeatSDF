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
from trainers import analyticSDFs, RHS #, load_siren_sdf
from models.borrowed_PINN_model import DR
from scipy.spatial import cKDTree
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np
from trainers.utils.vis_utils import imf2mesh
from utils import load_imf
from notebooks import evaluation_surfaces

class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        self.net =  DR(cfg.models.decoder.dim,cfg.models.decoder.arch,cfg.models.decoder.out_dim, cfg.models.decoder.bc) #, cfg.models.decoder.bc) 
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
        
    def update(self,  cfg, writer, epoch, func,pts, f_net, *args, **kwargs):
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
        dim_out = cfg.models.decoder.out_dim
        
        if (func != None):
            phi_func = func
        else: 
            phi_func = analyticSDFs.phi_func_sphere
        #phi_func = analyticSDFs.phi_func_sphere
        
        tau = 0.02
        def u_0(xyz): 
            ret_0 = torch.where(xyz[:,2] > 0, torch.ones_like(xyz[:,2]), torch.zeros_like(xyz[:,2]))
            ret =f_net(xyz).view(bs)#torch.where(torch.norm(xyz - tens([0.095, 0.35, -0.5]), dim = -1) < 0.1, torch.ones_like(xyz[:,2]), torch.zeros_like(xyz[:,2]))
            # 
            return ret
        if (dim_out == 1):
            if cfg.models.decoder.approach == "exact":
                
                xyz, xyz_org = evaluation_surfaces.fibonacci_sphere(bs) #fibonacci_sphere(bs) #val_uni_sphere(bs) 
                xyz = pts
                bs = xyz.shape[0]
                xyz = tens(xyz).reshape(bs, 3)
                #xyz_org = tens(xyz_org).reshape(bs, 3)
                u = self.net(xyz)
                phi = phi_func(xyz)
                #print(phi)
                #print(xyz)
                normal = gradient(phi, xyz)#/torch.norm(gradient(phi, xyz), dim = -1).view(xyz.shape[0], 1)
                d1u = manifold_gradient(u.reshape(bs), xyz, normal.cuda())
                f = (u_0(xyz))
                
                fu = torch.square(f) + 2*f*u.reshape(bs)
                
                #*weight.reshape(bs,1)
                loss = (torch.square(u).reshape(bs,1) + tau*torch.square(torch.norm(d1u, p = 2, dim=(-1))).reshape(bs,1) - fu.reshape(bs, 1)).mean()
            elif cfg.models.decoder.approach == "band":
                '''r = np.random.uniform(0.9, 1.1, bs)
                phi = np.random.uniform(0, 2*np.pi, bs)
                theta = np.random.uniform(0, np.pi, bs)
                i = 0
                xy = [None]*bs
                factor = [None]*bs
                while i < bs:
                    xy[i] = [np.sin(theta[i])*np.cos(phi[i]), np.sin(theta[i])*np.sin(phi[i]), np.cos(theta[i])]
                    factor[i] = 1/(r[i]**2*np.sin(theta[i]))
                    i += 1 
                factor = tens(factor)
                xy = tens(xy)
                '''
             
                phi = phi_func
                sample_size = 10000
                box_width = 2.15
                a = 2.1
                b = 1.1
                c = 0.8
                sample_l = tens(np.random.uniform(-a,a, (sample_size,1)))
                sample_w = tens(np.random.uniform(-b,b, (sample_size,1)))
                sample_h = tens(np.random.uniform(-c,c, (sample_size,1)))
                sample = torch.cat((sample_l, sample_w, sample_h), 1)
                
                #sample = tens(np.random.uniform(-box_width,box_width, (sample_size,3)))
                
                phi_sort, indices = torch.sort(torch.abs(phi(sample)), dim = 0)              
                t = torch.max((phi_sort < 0.1).nonzero(as_tuple=True)[0])
                indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
                xy = sample[indices[indices_to_keep]].reshape(t+1, 3)
                true_bs = xy.shape[0]
                
                while (true_bs < bs):
                    sample = tens(np.random.uniform(-box_width,box_width, (sample_size,3)))
                    phi_sort, indices = torch.sort(torch.abs(phi(sample)), dim = 0)
                    #sample = sample[indices].squeeze()
                    
                    if((phi_sort < 0.1).nonzero(as_tuple=True)[0].shape[0] > 0 and (phi_sort < 0.1).nonzero(as_tuple=True)[0].shape[0] < sample_size):
                        t = torch.max((phi_sort < 0.1).nonzero(as_tuple=True)[0])
                        indices_to_keep = [i for i in range(phi_sort.shape[0]) if i <= t.item()]
                        sample = sample[indices[indices_to_keep]].reshape(t+1, 3)
                        
                        xy= torch.cat((xy, sample), 0)
                        true_bs = xy.shape[0]
                bs = true_bs   
            
                '''
                level_set_count = 25
                random_treshold = np.random.uniform(-0.1, 0.1, level_set_count)
                xy = tens([])
                i = 0
                while (i < level_set_count):
                    #res = 10 seems good
                    sample = tens(imf2mesh(phi_func,threshold=random_treshold[i], verbose=False, res = 10, normalize=True, bound = 1.2).vertices)
                    xy= torch.cat((xy, sample), 0)
                    i += 1
                bs = xy.shape[0]
                ''''''
                xy = xy.cpu().detach()
                tree = cKDTree(xy)
                new_points = tens([])
                for point in xy:
                    # 1. Nächste Nachbarn finden
                    _, idxs = tree.query(point, k=10)  # Verwende 6 Nachbarn

                    neighbors = xy[idxs]
                    
                    # 2. Schwerpunkt der Nachbarn berechnen
                    centroid = torch.mean(neighbors, axis=0)

                    # 3. Punkt in Richtung des Schwerpunkts bewegen
                    direction = centroid - point
                    step = direction / np.linalg.norm(direction) * 0.1
                    new_point = point + step
                    new_points = torch.cat((new_points, new_point.cuda().reshape(1, 3)), 0)
                    #new_points.append(new_point)
                xy = new_points
                '''
                
                writer.add_scalar('train/batch_size', bs, epoch)
                #xy = np.random.uniform(-1.1,1.1, (bs, 3))
                #xy = tens(xy)
                u = self.net(xy)
                phi = phi_func(xy)
               
                grad_phi = gradient(phi, xy).view(
                    bs, 3)
                normal = grad_phi/torch.norm(grad_phi, dim = -1).view(bs, 1)
                
                d1u = manifold_gradient(u.reshape(bs), xy, normal.cuda())

                f = RHS.RHS_1(xy, phi).reshape(bs)
                fu = f*u.reshape(bs)
#######################################################   
                x = ((torch.abs(phi) - 0.1) / (0.075 - 0.1)).reshape(bs,1) 
                smooth_cut_off = torch.where(torch.abs(phi).reshape(bs,1) < 0.075,  torch.ones(bs,1).cuda(), 3*x**2 - 2*x**3).reshape(bs,1)*torch.norm(grad_phi, dim = -1).view(bs, 1)
                f = (u_0(xyz))
                fu = 1/2*torch.square(f) + f*u.reshape(bs)
                
                #*weight.reshape(bs,1)
                loss = (((1/2)*torch.square(torch.square(u)).reshape(bs,1) - (tau/2)*torch.square(torch.norm(d1u, p = 2, dim=(-1))).reshape(bs,1) - fu.reshape(bs, 1))).mean()
            
            elif cfg.models.decoder.approach == "PINN":
                xyz, xyz_org = evaluation_surfaces.fibonacci_sphere(bs) 
                xyz = tens(xyz).reshape(bs, 3)
                xyz_org = tens(xyz_org).reshape(bs, 3)
                u = self.net(xyz)
                phi = phi_func(xyz)
                normal = gradient(phi, xyz)/torch.norm(gradient(phi, xyz), dim = -1).view(xyz.shape[0], 1)
                d2u = lapl_beltrami(u.reshape(bs), xyz, normal.cuda())

                f = RHS.RHS_1(xyz, phi).reshape(bs)

                #weight = analyticSDFs.weightFEM(xyz_org)
                loss = ((torch.square(d2u.reshape(bs,1)- f.reshape(bs,1)).reshape(bs,1))).mean()
                #print(((torch.square(torch.norm(d2u, p = 2, dim=(-1)) - f).reshape(bs,1))*weight.reshape(bs,1)).shape)
            else: return print("Approach not implemented. Approach must be \"exact\" or \"band\"!" )
        if (dim_out == 3):
            m_grad_0 = manifold_gradient(u[:,0], xy, normal.cuda())
            hess1 = manifold_jacobian(m_grad_0, xy, normal)
            m_grad_1 = manifold_gradient(u[:,1], xy, normal.cuda())
            hess2 = manifold_jacobian(m_grad_1, xy, normal)
            m_grad_2 = manifold_gradient(u[:,2], xy, normal.cuda())
            hess3 = manifold_jacobian(m_grad_2, xy, normal)
            hess_sq = torch.square(torch.norm(hess1, "fro", dim = (1,2))) + torch.square(torch.norm(hess2, "fro", dim = (1,2))) + torch.square(torch.norm(hess3, "fro", dim = (1,2)))
            
            ### define force terms acting on manifold
            #F = 10*torch.ones(bs).cuda()
            def F(xy):
                vec = torch.where(xy[:,0] < 0., (-50)*torch.ones(bs).cuda(), (50)*torch.ones(bs).cuda())
                #vec = torch.where(torch.sqrt(torch.square(xy[:,1])) + torch.square(xy[:,0]) < 0.1, vec, torch.zeros_like(vec))
                #vec = torch.cat((torch.zeros((bs, 2)).cuda().reshape(bs, 1), vec.reshape(bs, 1)), 1)
                return vec
            fu = F(xy)*u[:,0]

            ### set variational boundary conditions (suboptimal)
            
            xy_bd = tens(np.random.uniform(-2,0, (bs, 2)))
            xy_bd1 = tens(np.random.uniform(-2,2, (bs, 1)))
            xy_bd = torch.cat((xy_bd, xy_bd1.reshape(bs, 1)), 1) #torch.zeros(bs).cuda().reshape(bs, 1)), 1)
            u_bd = self.net(xy_bd)
            bd_constr = torch.norm(u_bd, dim = 1, p = 2)# torch.where(xy_bd[:,0]< torch.zeros(bs).cuda(), torch.norm(u_bd, dim = 1), torch.zeros(bs).cuda())
            
            ### Isometry constraint (Id + Du)^t(Id + Du) = Id

            D1 = manifold_gradient(u[:,0], xy, normal).reshape(bs, dim_out,1)
            D2 = manifold_gradient(u[:,1], xy, normal).reshape(bs, dim_out,1)
            D3 = manifold_gradient(u[:,2], xy, normal).reshape(bs, dim_out,1)
            Du = torch.cat((D1,D2,D3), 2).transpose(1,2)
            DuId = torch.matmul(torch.eye(3,dim_out).cuda().repeat(bs, 1, 1), Du)
            val = torch.matmul(torch.transpose(Du, 1,2), Du) + DuId + torch.transpose(DuId, 1,2)
            isom_loss = torch.square(torch.norm((val), "fro", dim = (1,2) ))
            
            loss = (1/2*hess_sq + fu).mean()  + 10*isom_loss.mean() #+ 10000*bd_constr.mean()
        if not no_update:
            loss.backward()
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

    def validate(self,  cfg, writer, epoch, func,pts, f_net, *args, **kwargs):
        ### samling
        bs = cfg.data.train.batch_size
        #if points == None:
        xy = np.random.uniform(-2.1,2.1, (bs, 3))
        #else: xy = points
        xy = tens(xy)
        dim_out = cfg.models.decoder.out_dim
        
        if (func != None):
            phi_func = func
        else: 
            phi_func = analyticSDFs.phi_func_sphere
        #phi_func = analyticSDFs.phi_func_sphere
        
        tau = 0.02
        def u_0(xyz): 
            ret_0 = torch.where(xyz[:,2] > 0, torch.ones_like(xyz[:,2]), torch.zeros_like(xyz[:,2]))
            #
            ret =  f_net(xyz).view(bs)#torch.where(torch.norm(xyz - tens([0.095, 0.35, -0.5]), dim = -1) < 0.1, torch.ones_like(xyz[:,2]), torch.zeros_like(xyz[:,2]))
            return ret
        if (dim_out == 1):
            if cfg.models.decoder.approach == "exact":
                
                xyz, xyz_org = evaluation_surfaces.fibonacci_sphere(bs) #fibonacci_sphere(bs) #val_uni_sphere(bs) 
                xyz = pts
                bs = xyz.shape[0]
                xyz = tens(xyz).reshape(bs, 3)
                #xyz_org = tens(xyz_org).reshape(bs, 3)
                u = self.net(xyz)
                phi = phi_func(xyz)
                #print(phi)
                #print(xyz)
                normal = gradient(phi, xyz)#/torch.norm(gradient(phi, xyz), dim = -1).view(xyz.shape[0], 1)
                d1u = manifold_gradient(u.reshape(bs), xyz, normal.cuda())
                f = (u_0(xyz))
                
                fu = torch.square(f) + 2*f*u.reshape(bs)
                
                #*weight.reshape(bs,1)
                loss = (torch.square(u).reshape(bs,1) + tau*torch.square(torch.norm(d1u, p = 2, dim=(-1))).reshape(bs,1) - fu.reshape(bs, 1)).mean()
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
