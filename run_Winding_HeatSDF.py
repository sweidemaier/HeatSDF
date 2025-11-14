from train_heat import run_training as run_heat_training
from train_SDF_Winding import run_training as run_SDF_training
from trainers.standard_utils import update_config
import subprocess
from pathlib import Path

# Path to the config file
CONFIG_FILE = "configs/NeuralSDFs.yaml"
BASE_PATH = Path(__file__).parent.resolve()
REPO_PATH = BASE_PATH / "third_party" / "dipole-normal-prop"

cmd = [
    "python", "-u", str(REPO_PATH / "orient_pointcloud.py"),
    "--pc", str(BASE_PATH / "Example_Clouds/hand.xyz"),
    "--export_dir", str(BASE_PATH / "Dipole_oriented_Clouds/hand"),
    "--models", str(REPO_PATH / "pre_trained/hands2.pt"),
    str(REPO_PATH / "pre_trained/hands.pt"),
    str(REPO_PATH / "pre_trained/manmade.pt"),
    "--iters", "3",
    "--propagation_iters", "5",
    "--number_parts", "30",
    "--minimum_points_per_patch", "100",
    "--weighted_prop",
    "--estimate_normals",
    "--diffuse"
]

result = subprocess.run(cmd, cwd=str(REPO_PATH), capture_output=True, text=True)

# 3️⃣ Run the command
subprocess.run(cmd, check=True)

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
pt_path = str(BASE_PATH / "Dipole_oriented_Clouds/hand/final_result.xyz")
run_SDF_training(CONFIG_FILE, pt_path )
print("SDF Step completed. Training process finished.")

