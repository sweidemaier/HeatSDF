# HeatSDF
![We compute neural SDFs from unoriented point clouds (left) by first computing a small time step of heat flow (middle) and then using its gradient directions to solve for a neural SDF (right).](Teaser-1.png)
This repository contains the official code for the paper:

**"SDFs from Unoriented Point Clouds using Neural Variational Heat Distances"**  
📄 [arXiv:2504.11212](https://arxiv.org/abs/2504.11212)

---

## 🔥 Overview

**HeatSDF** is a neural framework that reconstructs Signed Distance Functions (SDFs) from **unoriented point clouds** using a novel variational approach based on heat distances.

## 🧠 Abstract

We propose a novel variational approach for computing neural Signed Distance Fields (SDF) from unoriented point clouds. To this end, we replace the commonly used eikonal equation with the heat method, carrying over to the neural domain what has long been standard practice for computing distances on discrete surfaces. This yields two convex optimization problems for whose solution we employ neural networks: We first compute a neural approximation of the gradients of the unsigned distance field through a small time step of heat flow with weighted point cloud densities as initial data. Then we use it to compute a neural approximation of the SDF. We prove that the underlying variational problems are well-posed. Through numerical experiments, we demonstrate that our method provides state-of-the-art surface reconstruction and consistent SDF gradients. Furthermore, we show in a proof-of-concept that it is accurate enough for solving a PDE on the zero-level set.

---

## 🛠 Installation
This repository provides a Anaconda environment, and requires NVIDIA GPU to run the optimization routine. The code is tested with the following main dependencies: cudatoolkit 11.0, pytorch 1.7.1, torchaudio 0.7.2, torchvision 0.8.2, scikit-learn 1.3.2, trimesh 4.5.3, ubuntu 20.04. 
The whole environment can be set-up using the following commands:

```bash
conda env create -f HeatSDF_env.yml
conda activate HeatSDF
```
## 🚀 Usage
To run the complete learning pipeline for both the heat method and SDF reconstruction, execute the following command:
```
python run_HeatSDF.py
```
This will start the training process, performing both the heat learning stage (to estimate gradient directions of the unsigned distance field) and the SDF learning stage (to reconstruct the signed distance function).

If you want to test the method on your own point clouds, simply modify the input paths in the relevant configuration file located in the config folder.

The config file allows you to adjust various settings, including data paths and hyperparameters. Especially, if you are only interested in an approximation of the SDF near the surface, use 
```
input.parameters.sampling: boxes
```
The results are saved in `/HeatSDF/logs/SDF<current_date>`, with the heat and SDF steps organized into their respective subfolders: `/HeatSDF/logs/SDF<current_date>/heat_step` and `/HeatSDF/logs/SDF<current_date>/SDF_step`.

In addition, we provide a variant of the method that uses **generalized winding numbers** for inside/outside separation. To run this version, execute:
```
python Winding_run_HeatSDF.py
```
Note: This method requires ground truth or estimated surface normals as input.



---
## ✍️ Citation
If you use this code or ideas from the paper, please cite:
``` bibtex
@article{HeatSDF,
  title={SDFs from Unoriented Point Clouds using Neural Variational Heat Distances},
  author={Weidemaier et al.},
  journal={arXiv preprint arXiv:2504.11212},
  year={2025}
}
