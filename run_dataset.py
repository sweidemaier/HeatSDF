import subprocess
import yaml
import os
import argparse
import time
print(os.getcwd())

# Path to the config file
CONFIG_FILE = "configs/recon/NeuralSDFs.yaml"
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

def update_config(param1):
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    # Update values
    config.input.point_path= str(param1)
         
    # Write back to the file
    with open(CONFIG_FILE, "w") as configfile:
  	    yaml.dump(config, configfile)
pointpaths = ["/home/weidemaier/HeatSDF/recon benchmark/lordquas_uniform.csv",
              "/home/weidemaier/HeatSDF/recon benchmark/gargoyle_uniform.csv",
              "/home/weidemaier/HeatSDF/recon benchmark/dc_uniform.csv",
              "/home/weidemaier/HeatSDF/recon benchmark/daratech_uniform.csv",
              "/home/weidemaier/HeatSDF/recon benchmark/anchor_uniform.csv"
]
for i in range(len(pointpaths)):
    update_config(pointpaths[i])
    subprocess.run(["python", "train_pipeline.py"])