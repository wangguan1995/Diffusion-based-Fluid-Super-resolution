# Diffusion-based-Fluid-Super-resolution
<br>

PyTorch implementation of 

**A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction** 

(Links to paper: <a href="https://www.sciencedirect.com/science/article/pii/S0021999123000670">Journal of Computational Physics</a> | <a href="https://arxiv.org/abs/2211.14680">arXiv</a>)

<div style style=”line-height: 25%” align="center">
<h3>Sample 1</h3>
<img src="https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/images/reconstruction_sample_01.gif">
<h3>Sample 2</h3>
<img src="https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/images/reconstruction_sample_02.gif">
</div>

## Overview
Denoising Diffusion Probablistic Models (DDPM) are a strong tool for data super-resolution and reconstruction. Unlike many other deep learning models which require a pair of low-res and high-res data for model training, DDPM is trained only on the high-res data. This feature is especially beneficial to reconstructing high-fidelity CFD data from low-fidelity reference, as it allows the model to be more independent of the low-res data distributions and subsequently more adaptive to various data patterns in different reconstruction tasks.

## Datasets
Datasets used for model training and sampling can be downloaded via the following links.

- High resolution data (ground truth for the super-resolution task) (<a href="https://figshare.com/ndownloader/files/39181919">link</a>)

- Low resolution data measured from random grid locations (input data for the super-resolution task) (<a href="https://figshare.com/ndownloader/files/39214622">link</a>)


## Running the Experiments
This code has been tested on the following environment:

```
python 3.8
pip install -r requirements.txt
```

Download the high res and low res data and save the data files to the subdirectory ``./data/``.

<!--
More details about how to run the experiments are coming soon.
-->

<b>Step 1 - Training</b>

- !!! You Need Fix: ``./configs/cylinder_re3900.yml`` and ``./train_ddpm/cylinder_re3900.yml``
- !!! You Need Care:``./.npz``, training will generate one npz file, replace the old ``./data/velocity_stat.npz``


In the subdirectory ``./train_ddpm/``, run:


```
# Re 3900 Cylinder Slice, velocity x
python main.py --config cylinder_re3900_UNet.yml --exp ./experiments/cylinder_re3900_UNet/ --ni

# Re 3900 Cylinder, Slice,  Spatial Temperal Patches, velocity x ,UNet
python main.py --config cylinder_re3900_st_UNet.yml --exp ./experiments/cylinder_re3900_st_UNet/ --ni

# Re 1000 JHTDB Slice, velocity x
python main.py --config jhtdb_re1000.yml --exp ./experiments/jhtdb_re1000/ --doc ./ --ni

# Re 1000 JHTDB Slice Spatial Temperal Patches, velocity x
python main.py --config jhtdb_re1000_spatial_temperal.yml --exp ./experiments/jhtdb_re1000/ --doc ./weights/jhtdb_re1000/ --ni

# Re 1000 JHTDB Slice Spatial Temperal Patches, DIT model, velocity x
python main.py --config jhtdb_re1000_st_dit.yml --exp ./experiments/jhtdb_re1000_dit/ --doc ./weights/jhtdb_re1000_dit/ --ni

# Re 1000 2D turbulence
python main.py --config ./km_re1000_rs256_conditional.yml --exp ./experiments/km256/ --doc ./weights/km256/ --ni
```

The checkpoint of the trained model is by default saved at the following trajectory. You can atler the saving directory according to your need by changing the values of ``--exp`` and ``--doc``.

``.../Diffusion-based-Fluid-Super-resolution/train_ddpm/experiments/km256/logs/weights/km256/``

Note: If you prefer not to go through Step 1, we provide the following pretrained checkpoints to directly start from <b>Step 2</b>:
<ol type="1">
  <li>model without physics-informed conditioning input (<a href="https://figshare.com/ndownloader/files/40320733">link</a>)</li>
  <li>model with physics-informed conditioning input (<a href="https://figshare.com/ndownloader/files/39184073">link</a>)</li>
</ol>


<b>Step 2 - Evaluating</b>

Add the model checkpoint file (e.g., ``baseline_ckpt.pth``) from <b>Step 1</b> to the following directory.

``.../Diffusion-based-Fluid-Super-resolution/pretrained_weights/``



In the main directory of this repo, run:

```
# Re 3900 Cylinder Slice, velocity x
python main.py --config cylinder_re3900_UNet_1.yml --seed 1234 --sample_step 1 --t 240 --r 30

# Re 3900 Cylinder, Slice,  Spatial Temperal Patches, velocity x ,UNet
python main.py --config cylinder_re3900_st_UNet.yml --seed 1234 --sample_step 1 --t 240 --r 30

# Re 1000 JHTDB Slice, velocity x
python main.py --config jhtdb_re1000.yml --seed 1234 --sample_step 3 --t 240 --r 30

# Re 1000 JHTDB, Slice, Spatial Temperal Patches, velocity x
python main.py --config jhtdb_re1000_spatial_temperal.yml --seed 1234 --sample_step 3 --t 240 --r 30

# Re 1000 JHTDB, Slice, Spatial Temperal Patches, DIT model, velocity x
python main.py --config jhtdb_re1000_st_dit.yml --seed 1234 --sample_step 3 --t 240 --r 30

# Re 1000 2D turbulence
python main.py --config kmflow_re1000_rs256_sparse_recons.yml --seed 1234 --sample_step 1 --t 240 --r 30
```


## References
If you find this repository useful for your research, please cite the following work.
```
@article{shu2023physics,
  title={A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction},
  author={Shu, Dule and Li, Zijie and Farimani, Amir Barati},
  journal={Journal of Computational Physics},
  pages={111972},
  year={2023},
  publisher={Elsevier}
}
```


This implementation is based on / inspired by:

- [https://github.com/ermongroup/SDEdit](https://github.com/ermongroup/SDEdit) (SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations)
- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) (Denoising Diffusion Implicit Models)

