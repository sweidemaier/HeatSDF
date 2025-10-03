from train_heat import run_training as run_heat_training
from Winding_train_SDF import run_training as run_SDF_training
from trainers.standard_utils import update_config

# Path to the config file
CONFIG_FILE = "configs/NeuralSDFs.yaml"

# Step 1: Train with HeatStep
print("Running Heat Step 1...")
update_config(CONFIG_FILE, "trainers.HeatStep", create_logdir=True, use_farfield=False, tau=0.005)
run_heat_training(CONFIG_FILE)
print("Heat Step 1 completed. Continuing...")

# Step 2 (optional): Compute SDF using farfield — uncomment if needed
# print("Running Heat Step 2...")
# update_config(CONFIG_FILE, "trainers.HeatStep", create_logdir=True, use_farfield=True, tau=0.1)
# run_heat_training(CONFIG_FILE)
# print("Heat Step 2 completed. Continuing...")

# Step 3: Train with NeatSDF
print("Running SDF Step...")
update_config(CONFIG_FILE, "trainers.Winding_SDFStep")
run_SDF_training(CONFIG_FILE)
print("SDF Step completed. Training process finished.")