import os
import yaml
import time
import argparse
import importlib
import os.path as osp
from utils import AverageMeter, dict2namespace, update_cfg_hparam_lst
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from utils import load_imf
import numpy as np
from torch.utils.data import DataLoader
import csv
from trainers.utils.new_utils import tens
from trainers.helper import comp_heat_gradients 
from trainers.helper import inside_outside_torch
from helper import load_pts


from trainers.InsideOutside import inside_outside
import torch
def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Flow-based Point Cloud Generation Experiment')
    

    # distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()

    # parse config file
    with open("/home/weidemaier/HeatSDF/configs/recon/NeuralSDFs.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    config, hparam_str = update_cfg_hparam_lst(config, args.hparams)

    #  Create log_name
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    logname = config.log_name
    config.log_name = "logs/" + logname + "/SDF_step"
    config.save_dir = "logs/" + logname + "/SDF_step"
    config.log_dir = "logs/" + logname + "/SDF_step"

    os.makedirs(osp.join(config.log_dir, 'config'))
    with open(osp.join(config.log_dir, "config", "config.yaml"), "w") as outf:
        yaml.dump(config, outf)

    return args, config



def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True
    logname = cfg.log_name
    writer = SummaryWriter(log_dir=cfg.log_name)
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    start_epoch = 0
    start_time = time.time()
    if args.resume:
        if args.pretrained is not None:
            start_epoch = trainer.resume(args.pretrained)
        else:
            start_epoch = trainer.resume(cfg.resume.dir)

    # If test run, go through the validation loop first
    if args.test_run:
        trainer.save(epoch=-1, step=-1)
        val_info = trainer.validate(epoch=-1)

    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs + start_epoch))
    step = 0
    duration_meter = AverageMeter("Duration")
    loader_meter = AverageMeter("Loader time")
    best_val = np.Infinity
    
    points = load_pts(cfg).cuda()
    inner, outer, occ = inside_outside_torch(points, grid_size = cfg.input.parameters.box_count, dilate=True) 
    inner.requires_grad_(True)
    outer.requires_grad_(True)
    ### load networks
    near_net,cfg_near = load_imf(cfg.input.near_path, return_cfg=False)
    kappa = (3/5)*near_net(occ).max().cpu().detach().numpy()
 
    if (cfg.input.far_path != "None"):
        far_net,_ = load_imf(cfg.input.far_path, return_cfg=False)
    else: far_net = None
    n_inner, n_outer = comp_heat_gradients(inner, outer, near_net, far_net, kappa)

    valmin = 2
    valcount = 0   
    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):
        # train for one epoch
        loader_start = time.time()
        
        for batchnumber in range(1000):

            #print(points.shape)
            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)
            step = batchnumber + 1000 * epoch + 1
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
        val_loss = trainer.validate(cfg, points, near_net, far_net, writer, epoch, inner, outer, n_inner, n_outer, kappa, box_points = occ)['loss']
        
        if(val_loss < best_val): 
            trainer.save_best_val(epoch, step)
            best_val = val_loss
        trainer.sch.step(val_loss)
       
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step)

        if (epoch + 1) % int(cfg.viz.val_freq) == 0 and \
                int(cfg.viz.val_freq) > 0:
            val_info = trainer.validate(cfg, points, near_net, far_net, writer, epoch, inner, outer, n_inner, n_outer, kappa, box_points = occ)
            

        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, writer=writer)
        
    trainer.save(epoch=epoch, step=step)
    writer.close()


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
