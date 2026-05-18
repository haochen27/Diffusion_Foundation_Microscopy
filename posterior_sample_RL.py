"""
Posterior Sampling with Poisson Noise (Richardson-Lucy)
======================================================
DAPS-based posterior sampling using RL deconvolution for Poisson noise model.
Designed for microscopy images where photon counting noise is dominant.
"""

import json
import yaml
import torch
import numpy as np
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
import argparse
from typing import Dict, Optional

# Import utils modules
from utils.operators import GaussianBlur
from utils.mcmc_sampler import MCMCSampler
from utils.scheduler import DAPS
from utils.model_wrapper import MicroscopyDiffusionModel
from utils.metrics import compute_metrics


# ============================================================================
# Utility Functions (shared with posterior_sample.py)
# ============================================================================

def load_image(image_path, size=(512, 512)):
    """
    Load and preprocess image for diffusion model (matches training preprocessing)

    Normalization follows the same pipeline as training:
    - uint8 images: [0, 255] -> [0, 1] -> [-1, 1]
    - uint16 images: [0, 65535] -> [0, 1] -> [-1, 1]
    - float images: min-max normalize if > 1, then -> [-1, 1]

    Args:
        image_path: Path to image file (PNG, TIFF, etc.)
        size: Target size (height, width)

    Returns:
        Tensor in range [-1, 1] for model input, shape [1, 1, H, W]
    """
    img = Image.open(image_path)

    if img.mode != 'L' and img.mode != 'I;16':
        img = img.convert('L')

    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr_01 = (arr.astype(np.float32) / float(info.max)).clip(0, 1)
        arr_normalized = arr_01 * 2.0 - 1.0
    elif arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() > 1.0:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
        arr_normalized = arr.astype(np.float32) * 2.0 - 1.0
    else:
        arr_normalized = arr.astype(np.float32) * 2.0 - 1.0

    img_tensor = torch.from_numpy(arr_normalized).unsqueeze(0).unsqueeze(0)
    return img_tensor


def save_results(images_dict, output_path, add_labels=False):
    """
    Save side-by-side comparison of all provided images.
    All inputs expected in [-1, 1] range.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert all images to [0, 1] and collect in order
    panels = []
    labels = []
    for name, img in images_dict.items():
        if img is not None:
            panels.append((img * 0.5 + 0.5).clamp(0, 1))
            labels.append(name)

    if len(panels) == 0:
        print(f"[WARNING] No images to save")
        return

    side_by_side = torch.cat(panels, dim=2)  # concat along width
    save_image(side_by_side, str(output_path))
    print(f"[SAVE] Saved to: {output_path}")

    if add_labels:
        try:
            from PIL import ImageDraw, ImageFont
            img = Image.open(output_path)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            panel_width = img.width // len(panels)
            for i, label in enumerate(labels):
                draw.text((i * panel_width + 10, 10), label, fill='white', font=font)
            img.save(output_path)
        except Exception as e:
            print(f"[WARNING] Could not add labels: {e}")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# Main Sampling Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Posterior sampling with Poisson noise (Richardson-Lucy)')
    parser.add_argument('--config', type=str,
                       default='./config/posterior_sampling_RL.yaml',
                       help='Path to config file')

    # Optional overrides
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Override checkpoint path')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Override test directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Override number of samples')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device')
    parser.add_argument('--poisson_scale', type=float, default=None,
                       help='Override Poisson scale (single value, overrides config list)')
    parser.add_argument('--random_select', action='store_true',
                       help='Randomly select images instead of taking first n')

    args = parser.parse_args()

    # Load config file
    print("=" * 80)
    print("Posterior Sampling with Poisson Noise (Richardson-Lucy)")
    print("=" * 80)
    print(f"[CONFIG] Loading config from: {args.config}")
    config = load_config(args.config)

    # Apply command-line overrides
    if args.checkpoint:
        config['model']['checkpoint_path'] = args.checkpoint
    if args.test_dir:
        config['data']['test_dir'] = args.test_dir
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    if args.num_samples:
        config['data']['num_samples'] = args.num_samples
    if args.device:
        config['device']['device'] = args.device
    if args.random_select:
        config['data']['random_select'] = True

    # Extract config values
    checkpoint = config['model']['checkpoint_path']
    test_dir = config['data']['test_dir']
    base_output_dir = config['output']['save_dir']
    num_samples = config['data']['num_samples']
    random_select = config['data'].get('random_select', False)
    images_are_clean = config['data'].get('images_are_clean', True)
    target_images = config['data'].get('target_images', None)
    save_intermediate = config['output'].get('save_intermediate', False)
    save_comparison = config['output'].get('save_comparison', True)
    add_labels = config['output'].get('add_labels', False)
    per_image_subdir = config['output'].get('per_image_subdir', False)
    device = config['device']['device']

    # DAPS parameters
    daps_config = config['daps']
    annealing_steps = daps_config['annealing_steps']
    sigma_max = daps_config['sigma_max']
    sigma_min = daps_config['sigma_min']
    rho = daps_config.get('rho', -7.0)
    mcmc_steps = daps_config['mcmc_steps']
    mcmc_lr = daps_config['mcmc_lr']
    tau = daps_config['tau']
    num_ode_steps = daps_config.get('num_ode_steps', 1)
    sigma_bar = daps_config.get('sigma_bar', 0.5)

    # Operator parameters
    operator_config = config['operator']
    kernel_size = operator_config['kernel_size']
    blur_intensity = operator_config['intensity']

    # Poisson parameters
    poisson_config = config.get('poisson', {})
    if args.poisson_scale is not None:
        scales = [args.poisson_scale]
    else:
        scales = poisson_config.get('scales', [10])
    rl_steps = poisson_config.get('rl_steps', 1)
    rl_alpha = poisson_config.get('rl_alpha', 0.5)
    rl_lam_reg = poisson_config.get('rl_lam_reg', 1e-3)
    rl_eps = poisson_config.get('rl_eps', 1e-8)

    # Advanced options
    adv_config = config['advanced']
    seeds_list = adv_config.get('seeds', None)
    if seeds_list is None:
        seeds_list = [adv_config['seed']]
    # Master seed (for deterministic image selection when random_select=True)
    seed = adv_config.get('seed', seeds_list[0])
    verbose = adv_config['verbose']

    # Set random seed (controls image-selection RNG; per-sample seeding happens in the loop)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n[CONFIG] Parameters:")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Test dir: {test_dir}")
    print(f"  Base output dir: {base_output_dir}")
    print(f"  Num samples: {num_samples}")
    print(f"  Device: {device}")
    print(f"  Annealing steps: {annealing_steps}")
    print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
    print(f"  Sigma bar: {sigma_bar}")
    print(f"  Poisson scales: {scales}")
    print(f"  RL steps: {rl_steps}, alpha: {rl_alpha}, lam_reg: {rl_lam_reg}")
    print(f"  Seeds ({len(seeds_list)}): {seeds_list}")
    if target_images:
        print(f"  Target images (stems): {target_images}")
    print(f"  Per-image subdir: {per_image_subdir}")

    # Load model
    model = MicroscopyDiffusionModel(checkpoint, device=device)

    # Get test images
    test_dir = Path(test_dir)
    all_image_files = sorted(list(test_dir.glob('*.png')))
    # Also check for tiff files common in microscopy
    all_image_files += sorted(list(test_dir.glob('*.tif')))
    all_image_files += sorted(list(test_dir.glob('*.tiff')))

    if not all_image_files:
        print(f"[ERROR] No images found in {test_dir}")
        return

    print(f"[DATA] Found {len(all_image_files)} total images in test directory")

    # Optional filter by stem name (e.g., ["a_79", "m_81"])
    if target_images:
        target_set = set(target_images)
        filtered = [p for p in all_image_files if p.stem in target_set]
        missing = target_set - {p.stem for p in filtered}
        if missing:
            print(f"[WARNING] target_images not found in {test_dir}: {sorted(missing)}")
        image_files = filtered
    elif random_select:
        if num_samples > len(all_image_files):
            num_samples = len(all_image_files)
        import random
        random.seed(seed)
        image_files = random.sample(all_image_files, num_samples)
    else:
        image_files = all_image_files[:num_samples]

    print(f"[DATA] Using {len(image_files)} images")
    if not image_files:
        print(f"[ERROR] No images selected after filtering; exiting.")
        return

    # ===== Sweep over Poisson scales =====
    for scale in scales:
        print(f"\n{'=' * 80}")
        print(f"[SCALE] Poisson scale = {scale}")
        print(f"{'=' * 80}")

        # Create forward operator (no Gaussian noise - Poisson only)
        operator = GaussianBlur(
            kernel_size=kernel_size,
            intensity=blur_intensity,
            device=device,
            sigma=0.0  # No Gaussian noise
        )
        # Set Poisson flags
        operator.use_poisson = True
        operator.poisson_scale = scale
        print(f"[OPERATOR] Gaussian blur: kernel_size={kernel_size}, intensity={blur_intensity}")
        print(f"[OPERATOR] Poisson noise: scale={scale}")

        # Create DAPS sampler (two-stage: RL above 1/SNR, Gaussian MCMC below)
        sampler = DAPS(
            annealing_steps=annealing_steps,
            mcmc_steps=mcmc_steps,
            mcmc_lr=mcmc_lr,
            tau=tau,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            num_ode_steps=num_ode_steps,
            sigma_bar=sigma_bar,
            rho=rho,
            rl_steps=rl_steps,
        )

        # Process each image (with seed sweep per image)
        scale_metrics = []
        for idx, image_path in enumerate(image_files):
            print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_path.name}")

            # Per-(image, scale) output directory
            if per_image_subdir:
                output_dir = Path(base_output_dir) / image_path.stem / f"scale_{int(scale)}"
            else:
                output_dir = Path(base_output_dir) / f"scale_{int(scale)}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load image (normalized to [-1, 1])
            loaded_image = load_image(image_path).to(device)

            per_seed_metrics = []
            for seed_i in seeds_list:
                print(f"  [SEED {seed_i}] sampling...")
                # Reseed for this run (controls Poisson noise + DAPS initial sample + MCMC noise)
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_i)

                # Create Poisson measurement
                if images_are_clean:
                    with torch.no_grad():
                        measurement = operator.measure_poisson(loaded_image, scale=scale)
                        measurement = torch.clamp(measurement, -1.0, 1.0)
                    input_for_display = measurement
                else:
                    measurement = torch.clamp(loaded_image, -1.0, 1.0)
                    input_for_display = loaded_image

                # Get initial sample from prior
                x_start = sampler.get_start(1, model)

                # Perform posterior sampling with RL
                with torch.no_grad():
                    if save_intermediate:
                        reconstructed, trajectory = sampler.sample(
                            model, x_start, operator, measurement, verbose=verbose, save_trajectory=True
                        )
                        traj_dir = output_dir / 'intermediate' / f'seed_{seed_i:04d}_{image_path.stem}'
                        traj_dir.mkdir(parents=True, exist_ok=True)
                        save_steps = list(range(0, len(trajectory), max(1, len(trajectory) // 10)))
                        if len(trajectory) - 1 not in save_steps:
                            save_steps.append(len(trajectory) - 1)
                        for i in save_steps:
                            step_data = trajectory[i]
                            step_num = step_data['step']
                            sigma_val = step_data['sigma']
                            x0hat_img = (step_data['x0hat'] * 0.5 + 0.5).clamp(0, 1)
                            x0y_img = (step_data['x0y'] * 0.5 + 0.5).clamp(0, 1)
                            save_image(x0hat_img, str(traj_dir / f'step_{step_num:04d}_sigma_{sigma_val:.4f}_x0hat.png'))
                            save_image(x0y_img, str(traj_dir / f'step_{step_num:04d}_sigma_{sigma_val:.4f}_x0y.png'))
                        print(f"    [SAVE] {len(save_steps)} intermediate steps -> {traj_dir}")
                    else:
                        reconstructed = sampler.sample(
                            model, x_start, operator, measurement, verbose=verbose, save_trajectory=False
                        )

                # Save results
                output_path = output_dir / f'seed_{seed_i:04d}_{image_path.stem}.png'
                if save_comparison:
                    result_dict = {
                        'measurement': input_for_display[0],
                        'reconstructed': reconstructed[0]
                    }
                    if images_are_clean:
                        result_dict['GT'] = loaded_image[0]
                    save_results(result_dict, output_path, add_labels=add_labels)
                else:
                    recon_01 = (reconstructed[0] * 0.5 + 0.5).clamp(0, 1)
                    save_image(recon_01, str(output_path))
                    print(f"    [SAVE] {output_path}")

                # Compute metrics if GT available
                if images_are_clean:
                    recon_m = compute_metrics(reconstructed[0].cpu(), loaded_image[0].cpu())
                    input_m = compute_metrics(input_for_display[0].cpu(), loaded_image[0].cpu())
                    entry = {'seed': seed_i, 'image': image_path.name,
                             'recon_psnr': recon_m['psnr'], 'recon_ssim': recon_m['ssim'],
                             'input_psnr': input_m['psnr'], 'input_ssim': input_m['ssim']}
                    per_seed_metrics.append(entry)
                    scale_metrics.append(entry)
                    print(f"    Input PSNR: {input_m['psnr']:.2f} dB | SSIM: {input_m['ssim']:.4f}")
                    print(f"    Recon PSNR: {recon_m['psnr']:.2f} dB | SSIM: {recon_m['ssim']:.4f}")

            # Save per-(image, scale) metrics summary
            if per_seed_metrics:
                avg = {k: float(np.mean([m[k] for m in per_seed_metrics]))
                       for k in ['recon_psnr', 'recon_ssim', 'input_psnr', 'input_ssim']}
                std = {k: float(np.std([m[k] for m in per_seed_metrics]))
                       for k in ['recon_psnr', 'recon_ssim', 'input_psnr', 'input_ssim']}
                summary = {
                    'method': f'Diffusion_RL_scale_{int(scale)}',
                    'image': image_path.name,
                    'num_seeds': len(per_seed_metrics),
                    'per_seed': per_seed_metrics,
                    'average': avg,
                    'std': std,
                }
                metrics_path = output_dir / 'metrics_summary.json'
                with open(metrics_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"  [METRICS] {image_path.stem} scale={int(scale)} "
                      f"avg recon PSNR={avg['recon_psnr']:.2f}±{std['recon_psnr']:.2f} dB | "
                      f"SSIM={avg['recon_ssim']:.4f}±{std['recon_ssim']:.4f}")
                print(f"  [METRICS] saved -> {metrics_path}")

        # Scale-wide aggregate summary (across all images, all seeds)
        if scale_metrics:
            agg_dir = Path(base_output_dir) / f"scale_{int(scale)}_summary"
            agg_dir.mkdir(parents=True, exist_ok=True)
            avg = {k: float(np.mean([m[k] for m in scale_metrics]))
                   for k in ['recon_psnr', 'recon_ssim', 'input_psnr', 'input_ssim']}
            agg = {'method': f'Diffusion_RL_scale_{int(scale)}',
                   'num_entries': len(scale_metrics),
                   'per_run': scale_metrics, 'average': avg}
            with open(agg_dir / 'metrics_summary.json', 'w') as f:
                json.dump(agg, f, indent=2)
            print(f"\n{'='*60}")
            print(f"  [SCALE {int(scale)}] Input Avg PSNR: {avg['input_psnr']:.2f} dB | SSIM: {avg['input_ssim']:.4f}")
            print(f"  [SCALE {int(scale)}] Recon Avg PSNR: {avg['recon_psnr']:.2f} dB | SSIM: {avg['recon_ssim']:.4f}")
            print(f"  Aggregate metrics: {agg_dir / 'metrics_summary.json'}")
            print(f"{'='*60}")

    print(f"\n[COMPLETE] All Poisson RL experiments completed!")
    print(f"[COMPLETE] Results in: {base_output_dir}")


if __name__ == '__main__':
    main()

