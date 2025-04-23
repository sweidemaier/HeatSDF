import os
import yaml
import argparse
import importlib
import os.path as osp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def dict2namespace(config):
    if isinstance(config, argparse.Namespace):
        return config
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_imf_PDE(log_path, config_fpath=None, ckpt_fpath=None,
             epoch=None, verbose=False,
             return_trainer=False, return_cfg=False):
    # Load configuration
    if config_fpath is None:
        config_fpath = osp.join(log_path, "config", "config.yaml")
    with open(config_fpath) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.Loader))
    cfg.save_dir = "logs"

    # Load pretrained checkpoints
    ep2file = {}
    last_file, last_ep = osp.join(log_path, "latest.pt"), -1
    if ckpt_fpath is not None:
        last_file = ckpt_fpath
    else:
        ckpt_path = osp.join(log_path, "checkpoints")
        if osp.isdir(ckpt_path):
            for f in os.listdir(ckpt_path):
                if not f.endswith(".pt"):
                    continue
                ep = int(f.split("_")[1])
                if verbose:
                    print(ep, f)
                ep2file[ep] = osp.join(ckpt_path, f)
                if ep > last_ep:
                    last_ep = ep
                    last_file = osp.join(ckpt_path, f)
            if epoch is not None:
                last_file = ep2file[epoch]
    print(last_file)

    trainer_lib = importlib.import_module("trainers.SurfaceEigenvalues")
    trainer = trainer_lib.Trainer(cfg, None)
    trainer.resume(last_file)
    
    if return_trainer:
        return trainer, cfg
    else:
        imf = trainer.net
        del trainer
        return imf, cfg

def load_imf(log_path, config_fpath=None, ckpt_fpath=None,
             epoch=None, verbose=False,
             return_trainer=False, return_cfg=False):
    # Load configuration
    if config_fpath is None:
        config_fpath = osp.join(log_path, "config", "config.yaml")
    with open(config_fpath) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.Loader))
    cfg.save_dir = "logs"

    # Load pretrained checkpoints
    ep2file = {}
    last_file, last_ep = osp.join(log_path, "latest.pt"), -1
    if ckpt_fpath is not None:
        last_file = ckpt_fpath
    else:
        ckpt_path = osp.join(log_path, "checkpoints")
        if osp.isdir(ckpt_path):
            for f in os.listdir(ckpt_path):
                if not f.endswith(".pt"):
                    continue
                ep = int(f.split("_")[1])
                if verbose:
                    print(ep, f)
                ep2file[ep] = osp.join(ckpt_path, f)
                if ep > last_ep:
                    last_ep = ep
                    last_file = osp.join(ckpt_path, f)
            if epoch is not None:
                last_file = ep2file[epoch]
    print(last_file)

    trainer_lib = importlib.import_module("trainers.Points2unsignedDF")
    trainer = trainer_lib.Trainer(cfg, None)
    trainer.resume(last_file)
    
    if return_trainer:
        return trainer, cfg
    else:
        imf = trainer.net
        del trainer
        return imf, cfg

def load_imf_quad(log_path, config_fpath=None, ckpt_fpath=None,
             epoch=None, verbose=False,
             return_trainer=False, return_cfg=False):
    # Load configuration
    if config_fpath is None:
        config_fpath = osp.join(log_path, "config", "config.yaml")
    with open(config_fpath) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.Loader))
    cfg.save_dir = "logs"

    # Load pretrained checkpoints
    ep2file = {}
    last_file, last_ep = osp.join(log_path, "latest.pt"), -1
    if ckpt_fpath is not None:
        last_file = ckpt_fpath
    else:
        ckpt_path = osp.join(log_path, "checkpoints")
        if osp.isdir(ckpt_path):
            for f in os.listdir(ckpt_path):
                if not f.endswith(".pt"):
                    continue
                ep = int(f.split("_")[1])
                if verbose:
                    print(ep, f)
                ep2file[ep] = osp.join(ckpt_path, f)
                if ep > last_ep:
                    last_ep = ep
                    last_file = osp.join(ckpt_path, f)
            if epoch is not None:
                last_file = ep2file[epoch]
    print(last_file)

    trainer_lib = importlib.import_module("trainers.Points2SDF_clone")
    trainer = trainer_lib.Trainer(cfg, None)
    trainer.resume(last_file)
    
    if return_trainer:
        return trainer, cfg
    else:
        imf = trainer.net
        del trainer
        return imf, cfg

def parse_hparams(hparam_lst):
    print("=" * 80)
    print("Parsing:", hparam_lst)
    out_str = ""
    out = {}
    for i, hparam in enumerate(hparam_lst):
        hparam = hparam.strip()
        k, v = hparam.split("=")[:2]
        k = k.strip()
        v = v.strip()
        print(k, v)
        out[k] = v
        out_str += "%s=%s_" % (k, v.replace("/", "-"))
    print(out)
    print(out_str)
    print("=" * 80)
    return out, out_str


def update_cfg_with_hparam(cfg, k, v):
    k_path = k.split(".")
    cfg_curr = cfg
    for k_curr in k_path[:-1]:
        assert hasattr(cfg_curr, k_curr), "%s not in %s" % (k_curr, cfg_curr)
        cfg_curr = getattr(cfg_curr, k_curr)
    k_final = k_path[-1]
    assert hasattr(cfg_curr, k_final), \
        "Final: %s not in %s" % (k_final, cfg_curr)
    v_type = type(getattr(cfg_curr, k_final))
    setattr(cfg_curr, k_final, v_type(v))


def update_cfg_hparam_lst(cfg, hparam_lst):
    hparam_dict, hparam_str = parse_hparams(hparam_lst)
    for k, v in hparam_dict.items():
        update_cfg_with_hparam(cfg, k, v)
    return cfg, hparam_str

def write_obj(filepath, vertices, faces=None, uv=None, fvt=None, fnt=None, vertex_normals=None, mtl_path=None, mtl_name="material_0", precision=None):
    #borrowed from F.H.

    n_vertices = len(vertices)
    n_faces = len(faces) if faces is not None else 0
    n_vt = len(uv) if uv is not None else 0
    precision = precision if precision is not None else 16

    if (mtl_path is not None) and (uv is not None) and (fvt is None):
        print('WARNING: Material and uv provided, but no face texture index')

    if mtl_path is not None and n_faces == 0:
        print('WARNING: Material provided, but no face. Ignoring material.')

    with open(filepath,'w') as f:
        if n_faces > 0 and mtl_path is not None:
            mtl_filename = os.path.splitext(os.path.basename(mtl_path))[0]
            f.write(f'mtllib {mtl_path}\ng\n')

        f.write(f'# {n_vertices} vertices - {n_faces} faces - {n_vt} vertex textures\n')

        for i in range(n_vertices):
            f.write(f'v {" ".join([f"{coord:.{precision}f}" for coord in vertices[i]])}\n')

        if vertex_normals is not None:
            for i in range(len(vertex_normals)):
                f.write(f'vn {" ".join([f"{coord:.{precision}f}" for coord in vertex_normals[i]])}\n')

        if uv is not None:
            for i in range(len(uv)):
                f.write(f'vt {" ".join([f"{coord:.{precision}f}" for coord in uv[i]])}\n')

        if n_faces > 0:
            if mtl_path is not None:
                f.write(f'g {mtl_filename}_export\n')
                f.write(f'usemtl {mtl_name}\n')

            for j in range(n_faces):
                if fvt is not None and fnt is not None:
                    f.write(f'f {" ".join([f"{1+faces[j][k]:d}/{1+fvt[j][k]:d}/{1+fnt[j][k]:d}" for k in range(3)])}\n')

                elif fvt is not None:
                    f.write(f'f {" ".join([f"{1+faces[j][k]:d}/{1+fvt[j][k]:d}" for k in range(3)])}\n')

                elif fnt is not None:
                    f.write(f'f {" ".join([f"{1+faces[j][k]:d}//{1+fnt[j][k]:d}" for k in range(3)])}\n')

                else:
                    f.write(f'f {" ".join([str(1+tri) for tri in faces[j]])}\n')
