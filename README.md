# HeatSDF

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

To set up the environment:

```bash
conda env create -f HeatSDF_env.yml
conda activate HeatSDF
```
## 🚀 Usage
To run the complete learning pipeline for both the heat method and SDF reconstruction, execute the following command:
```
python run_pipeline.py
```
This will start the training process, performing both the heat learning stage (to estimate gradients of the unsigned distance field) and the SDF learning stage (to reconstruct the signed distance function).

If you want to test the method on your own point clouds, simply modify the input paths in the relevant configuration file located in the config folder.

The config file allows you to adjust various settings, including data paths and hyperparameters. Especially, if you are only interested in an approximation of the SDF near the surface, use 
```
input.parameters.sampling: boxes
```

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
