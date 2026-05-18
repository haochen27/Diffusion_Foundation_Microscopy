[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/)

# Microscopy Diffusion Prior and Diffusion-based Sampling

Unconditional diffusion prior for microscopy images, plus a DAPS-based posterior-sampling pipeline (Gaussian and Poisson/Richardson–Lucy) for inverse problems such as deblurring and denoising.

## Checkpoints

Pretrained UNet foundation checkpoint:
[Download (Google Drive)](https://drive.google.com/file/d/1Ux3-oHpUS5s-h9q5gFtQpcC0e6cl2CvP/view?usp=sharing)

Place the checkpoint under `microscopy_runs_unet_foundation/checkpoints/last.ckpt` (or update `model.checkpoint_path` in the relevant config).

## Features

- **DiT and UNet Backbones**: DiT-XL/8 and UNet, both targeting 512×512 microscopy images
- **EDM Scheduler**: Elucidating the Design Space of Diffusion Models (Karras et al.)
- **Unconditional Foundation Training**: Single-phase training on foundation microscopy data
- **Real-Time Monitoring**: Live loss plots, sample generation, metrics tracking
- **EMA Training**: Exponential moving average for stable convergence
- **Smart Checkpoint Management**: Custom epoch-based checkpointing
- **State Persistence**: Full resume capability
- **Lightning Integration**: DDP, mixed precision
- **Comprehensive Sampling System**: DDPM, DDIM, and EDM Euler unconditional sampling
- **Posterior Sampling (DAPS)**: Gaussian-noise inverse problems via Decoupled Annealing Posterior Sampling
- **Posterior Sampling with Poisson Noise (Richardson–Lucy)**: Two-stage data consistency tailored to photon-counting noise

## Usage

### 1. Test Setup
```bash
python test_setup.py
```

### 2. Train Model
```bash
# UNet foundation (current default)
python train.py --config config/microscopy_unet.yaml

# DiT foundation
python train.py --config config/microscopy.yaml
```

### 3. Resume Training
```bash
python train.py --config config/microscopy_unet.yaml --resume
```

### 4. Generate Unconditional Samples
```bash
# Basic sampling
python sample.py --config config/sampling.yaml

# Custom parameters
python sample.py --config config/sampling.yaml --num_samples 8 --method edm_euler --steps 100
```

### 5. Posterior Sampling — Gaussian Noise (DAPS)
```bash
python posterior_sample.py --config config/posterior_sampling.yaml
```

### 6. Posterior Sampling — Poisson Noise (Richardson–Lucy)
```bash
python posterior_sample_RL.py --config config/posterior_sampling_RL.yaml
```

## Training

### Unconditional Foundation
- **Objective**: Learn the general microscopy image distribution
- **Data**: 46K+ foundation samples (SR-Caco2 + FMD)
- **Architecture**: DiT-XL/8 or UNet (no conditioning)
- **Epochs**: 50 (configurable)

## Sampling System

The project includes a comprehensive sampling system for generating samples from trained models.

### Sampling Methods
- **DDPM**: Standard denoising diffusion probabilistic models
- **DDIM**: Denoising diffusion implicit models (faster, deterministic)
- **EDM Euler**: Elucidating the Design Space of Diffusion Models

### Features
- **Standalone Inference**: Generate samples independently from training
- **Batch Generation**: Efficient large-scale sampling
- **VAE Integration**: Automatic latent space handling (optional)
- **Flexible Configuration**: YAML-based parameter configuration

### Configuration
Edit `config/sampling.yaml` to customize:
- Sampling method and steps
- Output image size and format
- Device and performance settings

## Posterior Sampling for Inverse Problems

The repository ships two posterior-sampling pipelines built on the trained unconditional prior. Both use Decoupled Annealing Posterior Sampling (DAPS) with annealed Langevin MCMC data consistency.

### Gaussian Noise — `posterior_sample.py`
Standard DAPS with Gaussian-likelihood data consistency. Suitable for deblurring, denoising, and super-resolution when the noise model is approximately Gaussian.

Config: `config/posterior_sampling.yaml`
- Forward operator: Gaussian blur, downsample, additive Gaussian noise
- MCMC: Langevin updates against a Gaussian likelihood
- Solver: Tweedie or multi-step Euler for `x0hat`

### Poisson Noise — `posterior_sample_RL.py`
DAPS with two-stage data consistency tailored to photon-counting (Poisson) noise. The transition between stages is governed by the noise SNR and is handled automatically inside `utils/mcmc_sampler.py::sample_RL`:

- **Stage 1** (`sigma >= 1/sqrt(poisson_scale) = 1/SNR`): Richardson–Lucy multiplicative updates against the Poisson measurement.
- **Stage 2** (`sigma <  1/SNR`): Gaussian MCMC against a synthetic clean measurement `K(prev_state)`, i.e. the forward operator applied to the most recent `x0hat`.

Config: `config/posterior_sampling_RL.yaml`
- Forward operator: Gaussian blur (no additive Gaussian noise)
- Poisson `scales` sweep (e.g. `[1, 5, 10, 20, 100]`)
- RL parameters: `rl_steps`, `rl_alpha` (damping), `rl_lam_reg`, `rl_eps`

Both pipelines reuse the same prior checkpoint (`model.checkpoint_path`) and write side-by-side comparisons of `measurement / reconstruction / ground truth` to `output.save_dir`.

## Data Format

Expects processed microscopy data with manifest structure:
- Foundation: High-quality images
- Manifests: CSV files with metadata (id, image, dataset, channel_idx, norm_lo, norm_hi, ...)

Images are normalized to `[-1, 1]` for training.

## Data

The dataset used for training can be found at the following link:
[Microscopy Dataset](https://drive.google.com/drive/folders/1t-8SP_YxnNr99ELzuNWtRAws_CNlwPK0?usp=sharing)

## Citation

### DiT

```bibtex
@article{peebles2023scalable,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2023}
}
```

### EDM

```bibtex
@inproceedings{karras2022elucidating,
  title={Elucidating the Design Space of Diffusion-Based Generative Models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

### DAPS

```bibtex
@article{zhang2024improving,
  title={Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing},
  author={Zhang, Bingliang and Chu, Wenda and Berner, Julius and Meng, Chenlin and Anandkumar, Anima and Song, Yang},
  journal={arXiv preprint arXiv:2407.01521},
  year={2024}
}
```
