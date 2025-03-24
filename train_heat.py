import os
import yaml
import time
import torch
import argparse
import importlib
import os.path as osp
from utils import AverageMeter, dict2namespace, update_cfg_hparam_lst
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from utils import load_imf
import numpy as np
import csv
from trainers.utils.new_utils import tens
from trainers.helper import comp_weights
from notebooks import evaluation_surfaces

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
    with open("/home/weidemaier/PDE Net/NFGP/configs/recon/NeuralSDFs.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    config, hparam_str = update_cfg_hparam_lst(config, args.hparams)

    # Currently save dir and log_dir are the same
    if not hasattr(config, "log_dir"):
        #  Create log_name
        cfg_file_name = os.path.splitext(os.path.basename("configs/recon/NeuralSDFs.yaml"))[0]
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
        
    points -= np.mean(points, axis=0, keepdims=True)

    coord_max = np.amax(points)
    coord_min = np.amin(points)
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.    
    points = np.float32(points)
    #weights = np.loadtxt("most_recent_weights.out", delimiter=",")
    #weights = torch.tensor(weights)
    weights = comp_weights(points,cfg.input.parameters.epsilon, cfg.models.decoder.dim)
    #weights = 1/line_count*torch.ones(line_count -1,1)
    #np.savetxt('most_recent_weights.out', weights, delimiter=",")

    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):

        # train for one epoch
        loader_start = time.time()
        leng = 100
        for batchnumber in range(100):

            #print(points.shape)
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
                visualize = step % int(cfg.viz.viz_freq) == 0 and \
                            int(cfg.viz.viz_freq) > 0
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
        
        # Save first so that even if the visualization bugged,
        # we still have something
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step)

        if (epoch + 1) % int(cfg.viz.val_freq) == 0 and \
                int(cfg.viz.val_freq) > 0:
            val_info = trainer.validate(cfg, weights, points, writer,  epoch=epoch)
            

        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, writer=writer)
        #print("Current learning rate: ", opt.param_groups[0]["lr"])
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
