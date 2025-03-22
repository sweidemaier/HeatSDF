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
from trainers.helper import comp_weights, comp_heat_gradients
from notebooks import error_evals
from notebooks import error_utils
from trainers.InsideOutside import inside_outside

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
    with open("/home/weidemaier/PDE Net/NFGP/configs/recon/NeuralSDFs3.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    config, hparam_str = update_cfg_hparam_lst(config, args.hparams)

    # Currently save dir and log_dir are the same
    if not hasattr(config, "log_dir"):
        #  Create log_name
        cfg_file_name = os.path.splitext(os.path.basename("configs/recon/NeuralSDFs3.yaml"))[0]
        run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        post_fix = hparam_str + run_time

        config.log_name = "logs/%s_%s" % (cfg_file_name, post_fix)
        config.save_dir = "logs/%s_%s" % (cfg_file_name, post_fix)
        config.log_dir = "logs/%s_%s" % (cfg_file_name, post_fix)

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
    ### load input points
    with open(cfg.input.point_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        file = open(cfg.input.point_path)
        count = len(file.readlines()) -1
        points = [None]*count
        line_count = 0
        for row in csv_reader:
            if (line_count == 0):
                line_count += 1
            else:
                a = float(row[0])
                b = float(row[1])
                if (cfg.models.decoder.dim == 3):
                    c = float(row[2])
                points[line_count-1] = [a, b]
                if (cfg.models.decoder.dim == 3):
                    points[line_count-1] = [a, b, c]
                line_count += 1  
    ###normalize points, st. they are in [-1, 1]    
    points -= np.mean(points, axis=0, keepdims=True)
    coord_max = np.amax(points)
    coord_min = np.amin(points)
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.    
    points = np.float32(points)
    inner, outer, occ = inside_outside(points, grid_size = 16)
    #inner, outer, _ = inside_outside(occ, grid_size = 32)
    train_dataloader = DataLoader(points, shuffle=True, batch_size=5000, pin_memory=True)
    ### load networks
    near_path = cfg.input.near_path
    far_path = cfg.input.far_path
    near_net,_ = load_imf(near_path, return_cfg=False) #, ckpt_fpath = near_path)# + "/best.pt")
    far_net,_ = load_imf(far_path, return_cfg=False) #, ckpt_fpath = far_path)# + "/best.pt")
    n_inner, n_outer = comp_heat_gradients(inner, outer, near_net, far_net, gamma = 500)
    
    valmin = 2
    valcount = 0   
    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):
        iter_tr = iter(train_dataloader)
        # train for one epoch
        loader_start = time.time()
        
        for batchnumber in range(1000): #range(len(train_dataloader)): #

            #print(points.shape)
            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)
            step = batchnumber + 1000 * epoch + 1
            #= next(iter_tr).squeeze() #points 
            logs_info = trainer.update(cfg, input_points = points , near_net = near_net, far_net = far_net, epoch = epoch, step = step, gt_inner = inner, gt_outer = outer, n_inner = n_inner, n_outer = n_outer)
            
            if step % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
                duration = time.time() - start_time
                duration_meter.update(duration)
                start_time = time.time()
                print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loading [%3.2fs]"
                      " Loss %2.5f"
                      % (epoch, batchnumber, 1000, duration_meter.avg,
                         loader_meter.avg, logs_info['loss']))
                visualize = step % int(cfg.viz.viz_freq) == 0 and \
                            int(cfg.viz.viz_freq) > 0
                trainer.log_train(
                    logs_info,
                    writer=writer, epoch=epoch, step=step, visualize=False)

            # Reset loader time
            loader_start = time.time()
        val_loss = trainer.validate(cfg, points, near_net, far_net, writer, epoch, inner, outer, n_inner, n_outer)['loss']
        
        if(val_loss < best_val): 
            trainer.save_best_val(epoch, step)
            best_val = val_loss
        trainer.sch.step(val_loss)
        if(val_loss < valmin): 
            valmin = val_loss
            valcount = 0
        valcount += 1
        if(cfg.trainer.opt.lr <= 10**(-6)):
            if (valcount > 10):
                err1, err2, err3 = error_evals.eval(logname)
                with open("overview.txt", "a") as f:
                    f.write(str([logname, err1, err2, err3]))
                    f.write("\n")

                f = open("overview.txt","r")
                break
        # Save first so that even if the visualization bugged,
        # we still have something
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step)

        if (epoch + 1) % int(cfg.viz.val_freq) == 0 and \
                int(cfg.viz.val_freq) > 0:
            val_info = trainer.validate(cfg, points, near_net, far_net, writer, epoch, inner, outer, n_inner, n_outer)
            

        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, writer=writer)
        
        #print("Current learning rate: ", opt.param_groups[0]["lr"])
    err1, err2, err3 = error_evals.eval(logname)
    with open("overview.txt", "a") as f:
        f.write(logname, err1, err2, err3)

    f = open("overview.txt","r")
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
