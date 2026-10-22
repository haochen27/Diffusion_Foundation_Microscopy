# Diffusion-RL

**Richardson–Lucy deconvolution with a learned diffusion prior for photon-limited fluorescence microscopy.**

[![Python](https://img.shields.io/badge/Python-3-1f425f.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code for *Improving Richardson–Lucy Deconvolution with Diffusion Priors for Fluorescence Microscopy* (H. Chen and S. S. Howard, manuscript under review).

Diffusion-RL keeps the Poisson physics of classical Richardson–Lucy (RL) and adds a prior learned from real fluorescence images:

1. **Prior.** An unconditional diffusion model (EDM, U-Net) trained on fluorescence micrographs proposes a clean image at each noise level. It never sees a PSF or a photon count, so one prior works across microscopes.
2. **Data consistency.** A few RL updates correct that proposal against the measured photon counts and the known PSF.

Alternating the two reduces RL's noise amplification and keeps weak structures (thin filaments, mitochondria) that TV regularization smooths away, with the largest gains at low photon counts.

## What's in the repo

| Script | What it does |
|---|---|
| `posterior_sample_RL.py` | **Diffusion-RL** reconstruction (Poisson noise) |
| `baseline_deconvolution.py` | RL and TV-RL baselines (no diffusion prior) |
| `train.py` | Train the diffusion prior |
| `sample.py` | Draw unconditional samples from the prior |
| `posterior_sample.py` | Variant for Gaussian noise (DAPS with Langevin data consistency) |

Settings live in `config/`; core code is in `core/` (models, EDM scheduler, training) and `utils/` (operators, samplers, metrics).

## Setup

```bash
git clone https://github.com/haochen27/Diffusion_Foundation_Microscopy.git
cd Diffusion_Foundation_Microscopy
pip install -r requirements.txt
python test_setup.py   # quick import check
```

A CUDA GPU is recommended.

## Download the checkpoint and data

| | Link | Put it at |
|---|---|---|
| Pretrained prior (U-Net, ~150 MB) | [last.ckpt](https://drive.google.com/file/d/1Ux3-oHpUS5s-h9q5gFtQpcC0e6cl2CvP/view?usp=sharing) | `microscopy_runs_unet_foundation/checkpoints/last.ckpt` |
| Training data (processed SR-Caco-2, FMD, Neuronal Cells, RxRx1) | [Google Drive folder](https://drive.google.com/drive/folders/15z9U74yZGnWw90BSO-gA8xNEzRV-VG2B?usp=drive_link) | `../processed/foundation` |

You only need the checkpoint to run reconstructions. The training data is needed only to retrain the prior. Other locations work if you update `model.checkpoint_path` or `data.foundation_path` in the config.

## Quick start

**1. Reconstruct with Diffusion-RL.** Put clean test images (`.png`) in a folder and set `data.test_dir` in `config/posterior_sampling_RL.yaml` to that folder. The script blurs each image with the configured PSF, draws Poisson counts, and reconstructs.

```bash
python posterior_sample_RL.py

# a single photon budget instead of the sweep in the config
python posterior_sample_RL.py --poisson_scale 10
```

Each output is a `measurement | reconstruction | ground truth` panel, saved to `output.save_dir`.

**2. Run the baselines.** They read the same config (PSF, photon budgets, test folder):

```bash
python baseline_deconvolution.py --mode rl                   # TV-RL
python baseline_deconvolution.py --mode rl --tv_weight 0     # plain RL
```

**3. Train the prior** (optional):

```bash
python train.py --config config/microscopy_unet.yaml
python train.py --config config/microscopy_unet.yaml --resume   # continue a run
```

**4. Sample from the prior** (optional):

```bash
python sample.py --config config/sampling.yaml \
  --checkpoint microscopy_runs_unet_foundation/checkpoints/last.ckpt --num_samples 8
```

## Main settings

In `config/posterior_sampling_RL.yaml`:

| Key | Meaning |
|---|---|
| `operator.kernel_size`, `operator.intensity` | Gaussian PSF support and width (px) |
| `poisson.scales` | Photon budgets to test (mean-count scale α) |
| `poisson.rl_steps` | RL updates per diffusion step |
| `poisson.rl_alpha` | Damping of each RL update, in (0, 1] |
| `daps.annealing_steps` | Number of diffusion (noise-level) steps |

At high noise levels the data-consistency step uses RL against the Poisson counts. At low noise levels, below 1/√(Poisson scale), it switches to Gaussian MCMC (`utils/mcmc_sampler.py::sample_RL`).

## Data format

Training data follows this layout (the Drive folder above already matches it):

```
<foundation_path>/
└── <dataset_name>/
    ├── images/                # PNG, 16-bit, or .npz
    └── manifests/manifest.csv # id, image, dataset, channel_idx, norm_lo, norm_hi, ...
```

Images are normalized to `[-1, 1]` for training.

## Citation

If you use this code, please cite:

```bibtex
@misc{chen2026diffusionrl,
  title  = {Improving {Richardson--Lucy} Deconvolution with Diffusion Priors for Fluorescence Microscopy},
  author = {Chen, Hao and Howard, Scott S.},
  year   = {2026},
  note   = {Manuscript under review}
}
```

This work builds on [EDM](https://github.com/NVlabs/edm) (Karras et al., NeurIPS 2022) and [DAPS](https://github.com/zhangbingliang2019/DAPS) (Zhang et al., CVPR 2025); please cite them as well. The optional DiT backbone follows [Peebles & Xie](https://github.com/facebookresearch/DiT) (ICCV 2023).

## License

[MIT](LICENSE)
