import os
import yaml
import time
import argparse
import importlib
import os.path as osp
from utils import AverageMeter, dict2namespace, update_cfg_hparam_lst
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import numpy as np
import torch
from utils import load_imf

class PointCloud(Dataset):
    def __init__(self, point_cloud, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        
        print("Finished loading point cloud")

        coords = point_cloud
        self.coords = coords

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        
        #coords -= np.mean(coords, axis=0, keepdims=True)
        #if keep_aspect_ratio:
        #    coord_max = np.amax(coords)
        #    coord_min = np.amin(coords)
        #else:
        #    coord_max = np.amax(coords, axis=0, keepdims=True)
        #    coord_min = np.amin(coords, axis=0, keepdims=True)
#
#        self.coords = (coords - coord_min) / (coord_max - coord_min)
#        self.coords -= 0.5
#        self.coords *= 2.
       
       
        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        
        total_samples = self.on_surface_points

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points, replace =False)

        on_surface_coords = self.coords[rand_idcs]
        
        
        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1 

        coords = on_surface_coords
        
        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float()}
                                                              


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
    with open("configs/recon/create_neural_fields.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    config, hparam_str = update_cfg_hparam_lst(config, args.hparams)

    # Currently save dir and log_dir are the same
    if not hasattr(config, "log_dir"):
        #  Create log_name
        cfg_file_name = os.path.splitext(os.path.basename("configs/recon/create_neural_fields.yaml"))[0]
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
    nearby_net, set = load_imf(cfg.input.net_path, return_cfg=True)
    far_net, set1 = load_imf(cfg.input.far_net, return_cfg=True) 
            
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

    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs + start_epoch))
    step = 0
    duration_meter = AverageMeter("Duration")
    loader_meter = AverageMeter("Loader time")
    #####        
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
        points = np.asarray(points) 
    points -= np.mean(points, axis=0, keepdims=True)

    coord_max = np.amax(points)
    coord_min = np.amin(points)
    
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.  
   
    points = np.float32(points)
    print(line_count)
    
    train_dataloader = DataLoader(points, shuffle=False, batch_size=1000, pin_memory=True)
    
    ####
    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):
        iter_tr = iter(train_dataloader)
        
        # train for one epoch
        loader_start = time.time()
        leng = 1000
        for batchnumber in range(len(train_dataloader)):  
            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)
            step = batchnumber + leng * epoch + 1
            
            logs_info = trainer.update(next(iter_tr).squeeze(), nearby_net, far_net, cfg)#next(iter_tr).squeeze(), input_net, cfg) #next(iter(train_dataloader))[0]['coords'].squeeze()
            #next(iter_tr).squeeze()
            if step % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
                duration = time.time() - start_time
                duration_meter.update(duration)
                start_time = time.time()
                print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loading [%3.2fs]"
                      " Loss %2.5f"
                      % (epoch, batchnumber, len(train_dataloader), duration_meter.avg,
                         loader_meter.avg, logs_info['loss']))
                
                
            # Reset loader time
            loader_start = time.time()
        visualize = True
        val = trainer.validate(nearby_net, far_net, points, cfg)
        trainer.sch.step(val['loss'])
        trainer.log_train(
            val, 
            writer=writer, epoch=epoch, step=step, visualize=visualize)
        # Save first so that even if the visualization bugged,
        # we still have something
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step)

        if (epoch + 1) % int(cfg.viz.val_freq) == 0 and \
                int(cfg.viz.val_freq) > 0:
            val_info = trainer.validate(nearby_net, far_net, points, cfg, epoch=epoch)
                        
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
