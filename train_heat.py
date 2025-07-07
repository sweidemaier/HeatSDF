import os
import yaml
import time
import torch
import importlib
import numpy as np
import os.path as osp
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from trainers.helper import comp_weights, load_pts
from trainers.standard_utils import AverageMeter, dict2namespace



def get_args(input_config):
    # parse config file
    with open(input_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    
    #  Create log_name
    logname = config.log_name
    config.log_name = "logs/" + logname + "/heat_step"
    config.save_dir = "logs/" + logname + "/heat_step"
    config.log_dir = "logs/" + logname + "/heat_step"

    os.makedirs(osp.join(config.log_dir, 'config'))
    with open(osp.join(config.log_dir, "config", "config.yaml"), "w") as outf:
        yaml.dump(config, outf)

    return config



def main_worker(cfg):
    # basic setup
    cudnn.benchmark = True
    
    writer = SummaryWriter(log_dir=cfg.log_name)
    trainer_lib = importlib.import_module("trainers.HeatStep")
    print(trainer_lib)
    trainer = trainer_lib.Trainer(cfg)

    start_epoch = 0
    start_time = time.time()
    
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs + start_epoch))
    step = 0
    duration_meter = AverageMeter("Duration")
    loader_meter = AverageMeter("Loader time")
    best_val = np.Infinity

    ### load points, compute locally adaptive weight and move them to cuda
    points = load_pts(cfg)
    weights = comp_weights(points,cfg.input.parameters.epsilon, cfg.models.decoder.dim)
    points = torch.tensor(points).cuda()
    weights = torch.tensor(np.float32(weights)).cuda()

    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):
        # train for one epoch
        loader_start = time.time()
        leng = 500
        for batchnumber in range(leng):
            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)
            step = batchnumber + leng * epoch + 1
            ### evaluate loss
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
                    writer=writer, epoch=epoch, step=step)
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



def run_training(input_config):
    # collect config settings and start training of heat step
    cfg = get_args(input_config)
    print("Configuration:")
    print(cfg)
    main_worker(cfg)
