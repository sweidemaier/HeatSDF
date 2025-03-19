import os
import torch
import os.path as osp
from trainers.utils.diff_ops import gradient ,divergence
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.new_utils import tens
from models.borrowed_PINN_model import DR
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np
from trainers.utils.vis_utils import imf2mesh
from trainers import analyticSDFs


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
        
    def update(self,  cfg, writer, epoch,  *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        lamda = 10**3
        bs = 25000
        p = 8
        bs = cfg.data.train.batch_size
        xy = np.random.uniform(-1.3,1.3, (bs, 3))
        xy = tens(xy)
        u = self.net(xy)
        grad_u = gradient(u, xy)
        grad_u_norm = torch.norm(grad_u, dim = -1)
        off_surface_penalty = torch.exp(-lamda*torch.abs(u))

        #fix signs inside and outside the surface
        phi = np.random.uniform(0, 2*np.pi, 500)
        theta = np.random.uniform(0, np.pi, 500)
        
        inner = np.column_stack((0.01*np.sin(theta)*np.cos(phi), 0.01*np.sin(theta)*np.sin(phi), 0.01*np.cos(theta)))
        outer = np.column_stack((2.3*np.sin(theta)*np.cos(phi), 2.3*np.sin(theta)*np.sin(phi), 2.3*np.cos(theta)))
        
        inner = tens(inner)
        outer = tens(outer)
        
        scaling = 10
        outer_loss = ((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean() 
        inner_loss = ((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean()
        boundary_fix = inner_loss + outer_loss
#+ off_surface_penalty.mean()
        ### fix gradient direction
        grad_phi = gradient(analyticSDFs.comp_FEM(xy), xy).view(
                bs, 3)
        phi_normal = grad_phi/torch.norm(grad_phi, dim = -1).view(bs, 1)
        u_normal = grad_u / torch.norm(grad_u, dim = -1).view(bs, 1)
        normal_alignment = torch.zeros(bs)
        factor = 100
        if (epoch < 25):
            normal_alignment = torch.norm(u_normal - phi_normal, dim = -1)
            factor = 10
        loss = torch.abs(self.net(xy) - torch.norm(xy, dim = -1) - torch.ones_like(torch.norm(xy, dim = -1))).mean()#factor*torch.square(grad_u_norm - torch.ones(bs).cuda()).mean()+ boundary_fix.mean() + normal_alignment.mean()
        ### alternative: p-laplace
        #grad_u = gradient(u, xy)
        #p_lapl = divergence((grad_u_norm**(p-2)).reshape(bs,1)*grad_u, xy)
        #loss = torch.square(p_lapl + torch.ones(bs).cuda()).mean() #+ off_surface_penalty.mean()
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

    def validate(self, cfg, writer, epoch, *args, **kwargs):
        
        lamda = 10**2
        p = 8
        bs = cfg.data.train.batch_size
        xy = np.random.uniform(-2.5,2.5, (bs, 3))
        xy = tens(xy)
        u = self.net(xy)
        grad_u = gradient(u, xy)
        grad_u_norm = torch.norm(grad_u, dim = -1)
        off_surface_penalty = torch.exp(-lamda*torch.abs(u))

        #fix signs inside and outside the surface
        phi = np.random.uniform(0, 2*np.pi, 500)
        theta = np.random.uniform(0, np.pi, 500)
        
        inner = np.column_stack((0.01*np.sin(theta)*np.cos(phi), 0.01*np.sin(theta)*np.sin(phi), 0.01*np.cos(theta)))
        outer = np.column_stack((2.2*np.sin(theta)*np.cos(phi), 2.2*np.sin(theta)*np.sin(phi), 2.2*np.cos(theta)))
        
        inner = tens(inner)
        outer = tens(outer)
        
        scaling = 100
        outer_loss = ((torch.arctan((-1)*scaling*self.net(outer)-0.1) + np.pi/2)/np.pi).mean() 
        inner_loss = ((torch.arctan(scaling*self.net(inner)-0.1) + np.pi/2)/np.pi).mean()
        boundary_fix = inner_loss + outer_loss

        ### fix gradient direction
        grad_phi = gradient(analyticSDFs.comp_FEM(xy), xy).view(
                bs, 3)
        phi_normal = grad_phi/torch.norm(grad_phi, dim = -1).view(bs, 1)
        u_normal = grad_u / torch.norm(grad_u, dim = -1).view(bs, 1)
        normal_alignment = torch.norm(u_normal - phi_normal, dim = -1)
        
#+ off_surface_penalty.mean()
        loss = 10*torch.square(grad_u_norm - torch.ones(bs).cuda()).mean() + boundary_fix.mean() + normal_alignment.mean()
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
