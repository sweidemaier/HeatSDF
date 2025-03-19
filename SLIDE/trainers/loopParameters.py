import subprocess
import yaml
import os
import argparse
os.chdir("..")
print(os.getcwd())

# Path to the config file
CONFIG_FILE = "configs/recon/NeuralSDFs2.yaml"
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
    config.input.parameters.param1 = str(param1)
    config.input.parameters.param2 = str(param2)
    #config.step_wise_incr = str(param3)
    #config.input.point_path = str(param1)
    # Write back to the file
    with open(CONFIG_FILE, "w") as configfile:
  	    yaml.dump(config, configfile)

def run_script():
    subprocess.run(["python", "train_SDF2.py"]) 

param_string = [150, 200]
heat_string = ["/home/weidemaier/PDE Net/NFGP/logs/NeuralSDFs2_2025-Mar-12-09-31-59", "/home/weidemaier/PDE Net/NFGP/logs/NeuralSDFs2_2025-Mar-12-09-33-47"]

# Loop with different values
#for i in range(1):  # Example loop
new_param1 = 100 #10*10**(i+1)
for j in range(len(param_string)):
    new_param2 = param_string[j] #135 + 15*j
    #new_param1 = "/home/weidemaier/PDE Net/NFGP/sphere" +str(i+3) +".csv"
    #new_param2 = "/home/weidemaier/PDE Net/NFGP/logs/sphere_path" + str(i+1)
    #new_param3 = #"/home/weidemaier/PDE Net/NFGP/logs/sphere" + str(i+1)
    comment = 0
    update_config(new_param1, new_param2, comment)
    

    print(f"Running with param1={new_param1}, param2={new_param2}")
    run_script()

