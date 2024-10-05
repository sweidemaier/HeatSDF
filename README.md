# SLIDE
SIREN based Learning for Implicit Distance Estimation

# Usage:

### Conda-Environment:
- conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
- conda install pyyaml
- conda install tensorboard
- conda install -c conda-forge trimesh
- pip install git+https://github.com/marian42/mesh_to_sdf.git
- conda install tqdm

### Training:
To change between heat/ 2nd- step change in config/recon/create_neural_field.yaml: 
- in the category  trainer - type - trainers.w_normf_trainer or trainers.heat_trainer, where heat_trainer corresponds to the heat flow learning and w_normf_trainer corresponds to the SDF learning using the input normalfield from step 1
- change in category input desired parameters (for heat: epsilon and tau, for 2nd-step: net-paths and other parameters)
  
Then run "python train_heat.py configs/recon/create_neural_fields.yaml --hparams data.path=/home/.../sphere2.npy" or "python train_with_normalfield.py configs/recon/create_neural_fields.yaml --hparams data.path=/.../sphere2.npy"

### Some open corrections
- unused stuff
- table tex file


