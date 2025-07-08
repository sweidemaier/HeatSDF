import os
import yaml
import time
import torch
import importlib
import numpy as np
import os.path as osp
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from trainers.standard_utils import AverageMeter, dict2namespace, load_imf
from trainers.helper import inside_outside_SDF, load_pts



def get_args(input_config):
    # parse config file
    with open(input_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    
    #  Create log_name
    logname = config.log_name
    config.log_name = "logs/" + logname + "/SDF_step"
    config.save_dir = "logs/" + logname + "/SDF_step"
    config.log_dir = "logs/" + logname + "/SDF_step"

    os.makedirs(osp.join(config.log_dir, 'config'))
    with open(osp.join(config.log_dir, "config", "config.yaml"), "w") as outf:
        yaml.dump(config, outf)

    return config


def main_worker(cfg):
    # basic setup
    cudnn.benchmark = True

    writer = SummaryWriter(log_dir=cfg.log_name)
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg)

    start_epoch = 0
    start_time = time.time()

    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs + start_epoch))
    step = 0
    duration_meter = AverageMeter("Duration")
    loader_meter = AverageMeter("Loader time")
    best_val = np.Infinity
    
    #load pointcloud and compute boxgrid for effective sampling and inner/outer regions 
    points = torch.tensor(load_pts(cfg)).cuda()
    domain_bound = cfg.input.parameters.domain_bound
    inner, outer, occ = inside_outside_SDF(points, grid_size = cfg.input.parameters.box_count, dilate=True, dim = cfg.models.decoder.dim, bound=domain_bound) 
    
    inner.requires_grad_(True)
    outer.requires_grad_(True)

    ### load heat step networks
    near_net,_ = load_imf(cfg.input.near_path)
    print(near_net(torch.tensor([[1.,0.,0.], [0.,0.,1.]]).cuda()))
    if (cfg.input.far_path != "None"):
        far_net,_ = load_imf(cfg.input.far_path)
    else: far_net = None
    kappa = (3/5)*near_net(occ).max().cpu().detach().numpy()
    
    ### start actual training loop
    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):
        # train for one epoch
        loader_start = time.time()
        
        for batchnumber in range(1000):

            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)
            step = batchnumber + 1000 * epoch + 1
            ### evaluate loss
            logs_info = trainer.update(cfg, input_points = points , near_net = near_net, far_net = far_net, epoch = epoch, step = step, gt_inner = inner, gt_outer = outer, kappa = kappa, box_points = occ)
            
            if step % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
                duration = time.time() - start_time
                duration_meter.update(duration)
                start_time = time.time()
                print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loading [%3.2fs]"
                      " Loss %2.5f"
                      % (epoch, batchnumber, 1000, duration_meter.avg,
                         loader_meter.avg, logs_info['loss']))
                trainer.log_train(
                    logs_info,
                    writer=writer, epoch=epoch, step=step, visualize=False)

            # Reset loader time
            loader_start = time.time()
        val_loss = trainer.validate(cfg, points, near_net, far_net, writer, epoch, inner, outer, kappa, box_points = occ)['loss']
        
        if(val_loss < best_val): 
            trainer.save_best_val(epoch, step)
            best_val = val_loss
        trainer.sch.step(val_loss)
       
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step, vis = False)
        
    trainer.save(epoch=epoch, step=step, vis = True)
    writer.close()


def run_training(input_config):
    # collect config settings and start training of SDF step
    cfg = get_args(input_config)
    print("Configuration:")
    print(cfg)
    main_worker(cfg)
