import os
import yaml
import time
import argparse
import importlib
import os.path as osp
from utils import AverageMeter, dict2namespace, update_cfg_hparam_lst
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from trainers.helper import comp_weights, load_pts
import torch

def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Flow-based Point Cloud Generation Experiment')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

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
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    config, hparam_str = update_cfg_hparam_lst(config, args.hparams)
    
    #  Create log_name
    logname = config.log_name
    config.log_name = "logs/" + logname + "/heat_step"
    config.save_dir = "logs/" + logname + "/heat_step"
    config.log_dir = "logs/" + logname + "/heat_step"

    os.makedirs(osp.join(config.log_dir, 'config'))
    with open(osp.join(config.log_dir, "config", "config.yaml"), "w") as outf:
        yaml.dump(config, outf)

    return args, config



def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True
    
    writer = SummaryWriter(log_dir=cfg.log_name)
    print(cfg.trainer.type)
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    start_epoch = 0
    start_time = time.time()
    
    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs + start_epoch))
    step = 0
    duration_meter = AverageMeter("Duration")
    loader_meter = AverageMeter("Loader time")
    best_val = np.Infinity

    points = load_pts(cfg)

    weights = comp_weights(points,cfg.input.parameters.epsilon, cfg.models.decoder.dim)

    points = torch.tensor(points).cuda()
    weights = torch.tensor(np.float32(weights)).cuda()

    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):
        # train for one epoch
        loader_start = time.time()
        leng = 100
        for batchnumber in range(100):
            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)
            step = batchnumber + leng * epoch + 1
            logs_info = trainer.update(cfg, weights, points)
            
            if step % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
                duration = time.time() - start_time
                duration_meter.update(duration)
                start_time = time.time()
                print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loading [%3.2fs]"
                      " Loss %2.5f"
                      % (epoch, batchnumber, leng, duration_meter.avg,
                         loader_meter.avg, logs_info['loss']))
                trainer.log_train(
                    logs_info,
                    writer=writer, epoch=epoch, step=step, visualize=False)
            # Reset loader time
            loader_start = time.time()
        val_loss = trainer.validate(cfg, weights, points, writer, epoch)['loss']
        if(val_loss < best_val): 
            trainer.save_best_val(epoch, step)
            best_val = val_loss
        trainer.sch.step(val_loss)
        
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step)
        
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
