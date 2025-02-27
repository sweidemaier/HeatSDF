import subprocess
import yaml
import os
import argparse
os.chdir("..")
print(os.getcwd())

# Path to the config file
CONFIG_FILE = "configs/recon/NeuralSDFs3.yaml"
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

def update_config(param1, param2, param3):
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    # Update values
    config.input.near_net = str(param3)#parameters.param1 = str(param1)
    config.input.far_net = str(param2)# parameters.param2 = str(param2)
    config.input.point_path = str(param1)
    # Write back to the file
    with open(CONFIG_FILE, "w") as configfile:
  	    yaml.dump(config, configfile)

def run_script():
    subprocess.run(["python", "train_SDF3.py"]) 

# Loop with different values
for i in range(4):  # Example loop
    #new_param1 = 10*10**i
    for j in range(1):
    #    new_param2 = 15*(2*j+1)
        new_param1 = "/home/weidemaier/PDE Net/NFGP/sphere" +str(i+1) +".csv"
        new_param2 = "/home/weidemaier/PDE Net/NFGP/logs/sphere" + str(i+1)
        new_param3 = "/home/weidemaier/PDE Net/NFGP/logs/sphere" + str(i+1)
        update_config(new_param1, new_param2, new_param3)
    
        print(f"Running with param1={new_param1}, param2={new_param2}")
        run_script()

