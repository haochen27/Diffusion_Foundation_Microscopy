[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/)

# Microscopy Diffusion

This repository has two major parts:

1. **Microscopy Diffusion Prior** — unconditional diffusion training on microscopy data (DiT or UNet, EDM scheduler).
2. **Diffusion-based Sampling** — posterior-sampling pipelines built on the trained prior, for inverse problems with Gaussian or Poisson noise.

## Checkpoints

Pretrained UNet foundation checkpoint:
[Download (Google Drive)](https://drive.google.com/file/d/1Ux3-oHpUS5s-h9q5gFtQpcC0e6cl2CvP/view?usp=sharing)

Place it under `microscopy_runs_unet_foundation/checkpoints/last.ckpt` (or update `model.checkpoint_path` in the relevant config).

## Setup
```bash
pip install -r requirements.txt
python test_setup.py
```

---

# Part 1 — Microscopy Diffusion Prior

Unconditional diffusion training to learn the general microscopy image distribution.

## Features
- **DiT and UNet Backbones** at 512×512
- **EDM Scheduler** (Karras et al.)
- **Lightning DDP**, mixed precision (bf16 / fp16)
- **EMA** (decay 0.9995, starts at step 2000)
- **Epoch-based checkpointing** with full resume
- **Real-time monitoring**: live loss plots, sample generation, GPU metrics

## Training

```bash
# UNet foundation (current default)
python train.py --config config/microscopy_unet.yaml

# DiT foundation
python train.py --config config/microscopy.yaml

# Resume
python train.py --config config/microscopy_unet.yaml --resume
```

### Unconditional Foundation
- **Objective**: Learn the general microscopy image distribution
- **Data**: 46K+ foundation samples (SR-Caco2 + FMD)
- **Architecture**: DiT-XL/8 or UNet (no conditioning)
- **Epochs**: 50 (configurable)

## Unconditional Sampling

```bash
# Basic
python sample.py --config config/sampling.yaml

# Custom parameters
python sample.py --config config/sampling.yaml --num_samples 8 --method edm_euler --steps 100
```

### Methods
- **DDPM** — standard denoising diffusion probabilistic models
- **DDIM** — denoising diffusion implicit models (faster, deterministic)
- **EDM Euler** — Karras et al. EDM solver

## Data Format

Expects preprocessed data with manifest structure:
```
<root>/
├── dataset_name/
│   ├── images/          # PNG, 16-bit, or .npz files
│   └── manifests/
│       └── manifest.csv # Columns: id, image, dataset, channel_idx, norm_lo, norm_hi, ...
```
Images are normalized to `[-1, 1]` for training. Configs reference data via `data.foundation_path` and `data.degraded_path`.

Dataset download: [Microscopy Dataset (Google Drive)](https://drive.google.com/drive/folders/1t-8SP_YxnNr99ELzuNWtRAws_CNlwPK0?usp=sharing)

---

# Part 2 — Diffusion-based Sampling

Posterior-sampling pipelines built on the trained unconditional prior for inverse problems (deblurring, denoising, super-resolution). Both use Decoupled Annealing Posterior Sampling (DAPS) with annealed Langevin MCMC data consistency.

## Gaussian Noise — `posterior_sample.py`

Standard DAPS with Gaussian-likelihood data consistency. Suitable when the measurement noise is approximately Gaussian.

```bash
python posterior_sample.py --config config/posterior_sampling.yaml
```

Config: `config/posterior_sampling.yaml`
- Forward operator: Gaussian blur, downsample, additive Gaussian noise
- MCMC: Langevin updates against a Gaussian likelihood
- Solver: Tweedie or multi-step Euler for `x0hat`

## Poisson Noise (Richardson–Lucy) — `posterior_sample_RL.py`

DAPS with two-stage data consistency tailored to photon-counting (Poisson) noise. The transition is governed by the noise SNR and is handled automatically inside `utils/mcmc_sampler.py::sample_RL`:

- **Stage 1** (`sigma >= 1/sqrt(poisson_scale) = 1/SNR`): Richardson–Lucy multiplicative updates against the Poisson measurement.
- **Stage 2** (`sigma <  1/SNR`): Gaussian MCMC against a synthetic clean measurement `K(prev_state)`, i.e. the forward operator applied to the most recent `x0hat`.

```bash
python posterior_sample_RL.py --config config/posterior_sampling_RL.yaml
```

Config: `config/posterior_sampling_RL.yaml`
- Forward operator: Gaussian blur (no additive Gaussian noise)
- Poisson `scales` sweep (e.g. `[1, 5, 10, 20, 100]`)
- RL parameters: `rl_steps`, `rl_alpha` (damping), `rl_lam_reg`, `rl_eps`

## Common

Both pipelines reuse the same prior checkpoint (`model.checkpoint_path`) and write side-by-side comparisons of `measurement / reconstruction / ground truth` to `output.save_dir`.

---

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
