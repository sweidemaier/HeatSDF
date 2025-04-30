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

def update_config(param1, create_logdir = False):
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    # Update values
    config.trainer.type= str(param1)
    if create_logdir:
        run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        config.log_dir = "SDF" + run_time
        config.log_name = "SDF" + run_time
        #TODO Florine: pass the path to this script 
        config.input.near_path= str("/home/weidemaier/HeatSDF/logs/SDF" + run_time + "/heat_step")
        config.input.far_path= str("/home/weidemaier/HeatSDF/logs/SDF" + run_time + "/heat_step")      
    # Write back to the file
    with open(CONFIG_FILE, "w") as configfile:
  	    yaml.dump(config, configfile)
        
### runs both heat and SDF step and safes resulting networks in same logs folder
def run_script():
    update_config("trainers.Points2unsignedDF", True)
    subprocess.run(["python", "train_heat.py"])
    update_config("trainers.Hessian_Points2SDF")
    subprocess.run(["python", "train_SDF.py"])


run_script()