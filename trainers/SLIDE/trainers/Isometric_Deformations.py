import os
import torch
import importlib
import os.path as osp
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient, divergence
from trainers.utils.diff_ops import laplace
from trainers.utils.diff_ops import jacobian as jac
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.new_utils import tens

torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np
def assemble_grad(grad_x, grad_y, bs, grad_sq):
    i = 0
    vec = [None]*bs
    if(grad_sq == False):
        while i < bs:
            vec[i] = [[grad_x[0][i].item(), grad_y[0][i].item()],[ grad_x[1][i].item(),  grad_y[1][i].item()],[grad_x[2][i].item(), grad_y[2][i].item()]]
            i += 1
        vec = tens(vec)
    else: 
        while i < bs:
            a = grad_x[0][i].item()*grad_y[0][i].item() + grad_x[1][i].item()*grad_y[1][i].item() + grad_x[2][i].item()*grad_y[2][i].item()
            vec[i] = [[grad_x[0][i].item()**2 + grad_x[1][i].item()**2 + grad_x[2][i].item()**2, a], [a ,grad_y[0][i].item()**2 + grad_y[1][i].item()**2 + grad_y[2][i].item()**2]]
            i += 1
        vec = tens(vec)
    return vec
def assemble_grad2(grad_1, grad_2, grad_3, bs):
    i = 0
    vec = [None]*bs
    
    while i < bs:
        vec[i] = [[grad_1[i][0].item(),grad_1[i][1].item()],[ grad_2[i][0].item(),grad_2[i][1].item() ],[grad_3[i][0].item(), grad_3[i][1].item()]]
        i += 1
    vec = tens(vec)
    return vec
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
        
    def update(self, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()
        
        bs =10
        bd_bs = 10
        L = 1
        alpha = 1/12
        
        ### initial deformation
        def phi_A(x, y):

            phi = (1/torch.pi)*torch.sin(torch.pi*x) #torch.tensor(((1/torch.pi)*torch.sin(torch.pi*x), y, (1/torch.pi)*torch.sin(torch.pi*x)))
            return phi
        ### explicit programming of phia, since something with the function doesn't work
        def grad_phi_x_expl(xy):
            return torch.stack((torch.cos(torch.pi*xy[:,0]), torch.zeros_like(torch.cos(torch.pi*xy[:,0])), -torch.sin(torch.pi*xy[:,0]))).cuda()
        def grad_phi_y_expl(xy):
            return torch.stack((torch.zeros_like(torch.cos(torch.pi*xy[:,0])), torch.ones_like(torch.cos(torch.pi*xy[:,0])), torch.zeros_like(torch.cos(torch.pi*xy[:,0])))).cuda()
        def D2_phi_A_1(xy):
            vec = [None]*bs
            i = 0
            while i < bs:
                vec[i] = [[-np.sin(np.pi*xy[i,0].item()), 0], [0,0]]
                i += 1
            vec = tens(vec)
            return vec 
        def D2_phi_A_2(xy):
            vec = [None]*bs
            i = 0
            while i < bs:
                vec[i] = [[0, 0], [0,0]]
                i += 1
            vec = tens(vec)
            return vec 
        def D2_phi_A_3(xy):
            vec = [None]*bs
            i = 0
            while i < bs:
                vec[i] = [[-np.cos(np.pi*xy[i,0].item()), 0], [0,0]]
                i += 1
            vec = tens(vec)
            return vec 
            
        xy = np.random.uniform(0,L, (bs, 2))
        xy = tens(xy)
        
        grad_phi_A = assemble_grad(grad_phi_x_expl(xy), grad_phi_y_expl(xy), bs, False)
        #####needs proper assembling here
        
        g_A = assemble_grad(grad_phi_x_expl(xy), grad_phi_y_expl(xy), bs, True)
        det_g_A = torch.det(g_A)
        g_inv = torch.inverse(g_A)
        
        phi = self.net(xy)

        def F(x):
            return torch.tensor((0,0,-0.1)).cuda()
        #print(torch.matmul(torch.transpose(grad_phi_A, 0,1), grad_phi_A))
        #det = torch.det(torch.matmul(torch.transpose(grad_phi_A, 0, 1) ,grad_phi_A))
        #print(det)
        
        D2_phi_1 = jac(gradient(phi[:,0], xy), xy)
        D2_phi_2 = jac(gradient(phi[:,1], xy), xy)
        D2_phi_3 = jac(gradient(phi[:,2], xy), xy)
        
        #D2_phi_sq = torch.square(D2_phi_1) + torch.square(D2_phi_2) + torch.square(D2_phi_3)
        

	##neumann boundary loss
        x_bd = np.random.uniform(0,L, bd_bs)
        i = 0
        vec = [None]*bd_bs
        while i < bd_bs:
            vec[i] = [0, x_bd[i]]
            i += 1
        vec = tens(vec)
        bd_grad_1 = gradient(self.net(vec)[0], vec)
    
        bd_grad_2 = gradient(self.net(vec)[1], vec)
        bd_grad = torch.norm(bd_grad_1) + torch.norm(bd_grad_2)
    ### main
        grad_net_1 = gradient(phi[:,0], xy)
        grad_net_2 = gradient(phi[:,1], xy)
        grad_net_3 = gradient(phi[:,2], xy)
        grad_net = assemble_grad2(grad_net_1, grad_net_2, grad_net_3, bs)

        n_a = torch.cross(grad_phi_A[:,:,0], grad_phi_A[:,:,1])
        n_a = n_a /torch.norm(n_a)
        n_b = torch.cross(grad_net[:,:,0], grad_net[:,:,1])
        n_b = n_b/torch.norm(n_b)
        '''
        i = 0
        prod_b = [None]*bs
        prod_a = [None]*bs
        while i < bs:
            prod_b[i] = torch.matmul(D2_phi[i], n_b[i]).item()
            prod_a[i] = torch.matmul(D2_phi_A(xy)[i], n_a[i]).item()
            i += 1
        prod_a = tens(prod_a)
        prod_b = tens(prod_b)
        '''
        #shape_op_rel = g_inv*(torch.sub(prod_b, prod_a))

#        tr_shape_op_rel = torch.trace(shape_op_rel)
        ###

        tr_shape_rel = 0
        i = 0
        j = 0
        k = 0
        l = 0
        g_half = torch.sqrt(g_inv)
        res = torch.zeros(bs).cuda()
        fin_res = torch.zeros(bs).cuda()
        while i < 2:
            while j < 2: 
                while k < 2:
                    while l < 2:
                        mul_B = D2_phi_1[:,k,l]* n_b[:,0] + D2_phi_2[:,k,l]* n_b[:,1] + D2_phi_3[:,k,l]* n_b[:,2]
                        mul_A = torch.zeros(bs).cuda()
                        if (k == 0 & l == 0):
                            mul_A = (-torch.pi*torch.sin(torch.pi*xy[:,0]))*n_a[:,0] + (-torch.pi*torch.cos(torch.pi*xy[:,0]))*n_a[:,2]
                        res += g_half[:, i, k]*g_half[:, l, j]*(mul_B - mul_A)
                        l += 1        
                    k += 1
                res = torch.square(res)
                fin_res += res
                res = torch.zeros(bs).cuda()
                j += 1
            i += 1
        main_term = torch.zeros(bs).cuda()
        term_0 = torch.zeros(bs).cuda()
        term_1 = torch.zeros(bs).cuda()
        term_2 = torch.zeros(bs).cuda()
        i = 0
        j = 0
        k = 0
        l = 0

        while i < 2:
            while j < 2: 
                while k < 2:
                    while l < 2:
                        term_0 +=g_half[:, i, k]*g_half[:, l, j]*D2_phi_1[:,k,l]
                        term_1 +=g_half[:, i, k]*g_half[:, l, j]*D2_phi_2[:,k,l]
                        term_2 +=g_half[:, i, k]*g_half[:, l, j]*D2_phi_3[:,k,l]
                        l += 1        
                    k += 1
                j += 1
            i += 1
        term_0 = torch.square(term_0)
        term_1 = torch.square(term_1)
        term_2 = torch.square(term_2)
        main_term = term_0 + term_1 + term_2
        #print(main_term)
        i = 0
        while i < bs:
            D2_phi_1[i] *= -n_b[i,0]
            i += 1

        diff_1 = D2_phi_1 - D2_phi_A_1(xy)
        val_1 = torch.matmul(torch.matmul(g_half,diff_1), g_half)

        i = 0
        while i < bs:
            D2_phi_2[i] *= -n_b[i,1]
            i += 1
        diff_2 = D2_phi_2 - D2_phi_A_2(xy)
        val_2 = torch.matmul(torch.matmul(g_half,diff_2), g_half)

        i = 0
        while i < bs:
            D2_phi_3[i] *= -n_b[i,2]
            i += 1
        diff_3 = D2_phi_3 - D2_phi_A_3(xy)
        val_3 = torch.matmul(torch.matmul(g_half,diff_3), g_half)
  
        main_term = torch.square(torch.norm(val_1, "fro", dim = (1,2))) + torch.square(torch.norm(val_2, "fro", dim = (1,2))) +torch.square(torch.norm(val_3, "fro", dim = (1,2)))
        
    
        
        
        loss = alpha/2* (torch.sqrt(det_g_A)*(main_term)).mean() - (torch.sqrt(det_g_A)*(F(xy)[0]*phi[:,0] +F(xy)[1]*phi[:,1] +F(xy)[2]*phi[:,2])).mean() #+ bd_grad.mean()
            #alpha/2* D2_phi_sq.mean()

      
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

    def validate(self,  epoch, *args, **kwargs):
        '''
        
        bs =1500

        alpha = 1

        xy = np.random.uniform(0,L, (bs, 2))
        xy = tens(xy)

        u = self.net(xy)
        def F(x):
            return torch.ones((bs, 3))
    
        lapl_u = laplace(u, xy)
        lapl_u_sq = torch.square(lapl_u)
        print(lapl_u)
        loss = alpha/2* lapl_u_sq.mean()- (F*u).mean() '''
        return 1

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
