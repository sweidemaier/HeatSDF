from train_heat import run_training as run_heat_training
from train_SDF import run_training as run_SDF_training
from trainers.standard_utils import update_config

# Path to the config file
CONFIG_FILE = "configs/NeuralSDFs.yaml"

# Step 1: Train with HeatStep
update_config(CONFIG_FILE, "trainers.HeatStep", create_logdir=True, use_farfield=False, tau=0.005)
run_heat_training(CONFIG_FILE)

# Step 2 (optional): Compute SDF using farfield — uncomment if needed
# update_config(CONFIG_FILE, "trainers.HeatStep", create_logdir=True, use_farfield=True, tau=0.1)
# run_heat_training(CONFIG_FILE)

# Step 3: Train with NeatSDF
update_config(CONFIG_FILE, "trainers.SDFStep")
run_SDF_training(CONFIG_FILE)



