"""
Posterior Sampling for Microscopy Diffusion Models
====================================================
DAPS-based posterior sampling using trained UNet model with Gaussian blur degradation
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


# ============================================================================
# Utility Functions
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
    # Load image with PIL (handles 8-bit and 16-bit automatically)
    img = Image.open(image_path)
    
    if img.mode != 'L' and img.mode != 'I;16':  # L=8-bit gray, I;16=16-bit gray
        img = img.convert('L')  # Convert to grayscale if needed
    
    img = img.resize(size, Image.BILINEAR)
    
    # Convert to numpy array (preserves bit depth)
    arr = np.array(img)
    
    # Normalize based on dtype (matches training preprocessing in utils/processed_dataset.py)
    if np.issubdtype(arr.dtype, np.integer):
        # Integer type (uint8, uint16, etc.)
        info = np.iinfo(arr.dtype)
        # Normalize: [0, dtype.max] -> [0, 1] -> [-1, 1]
        arr_01 = (arr.astype(np.float32) / float(info.max)).clip(0, 1)
        arr_normalized = arr_01 * 2.0 - 1.0
    elif arr.dtype == np.float32 or arr.dtype == np.float64:
        # Float type - check if needs min-max normalization
        if arr.max() > 1.0:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
        # Convert [0, 1] -> [-1, 1]
        arr_normalized = arr.astype(np.float32) * 2.0 - 1.0
    else:
        # Default: assume [0, 1] range
        arr_normalized = arr.astype(np.float32) * 2.0 - 1.0
    
    # Convert to tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(arr_normalized).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def save_results(images_dict, output_path, add_labels=False):
    """
    Save side-by-side comparison: measurement | reconstructed
    
    Normalization flow:
    - Input: tensors in [-1, 1] range (model output)
    - Convert: [-1, 1] -> [0, 1] for display
    - Save: torchvision.save_image handles [0, 1] -> [0, 255] conversion
    
    Args:
        images_dict: Dictionary with 'measurement' and 'reconstructed' keys (in [-1, 1])
        output_path: Path to save the image
        add_labels: Whether to add text labels
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert from [-1, 1] to [0, 1] for saving
    measurement = images_dict.get('measurement')
    reconstructed = images_dict.get('reconstructed')
    
    if measurement is not None:
        # [-1, 1] -> [0, 1], then clip to ensure valid range
        measurement = (measurement * 0.5 + 0.5).clamp(0, 1)
    
    if reconstructed is not None:
        # [-1, 1] -> [0, 1], then clip to ensure valid range
        reconstructed = (reconstructed * 0.5 + 0.5).clamp(0, 1)
    
    # Create side-by-side comparison
    if measurement is not None and reconstructed is not None:
        # Stack horizontally: [measurement | reconstructed]
        side_by_side = torch.cat([measurement, reconstructed], dim=2)  # Concatenate along width
        # save_image converts [0, 1] -> [0, 255] automatically
        save_image(side_by_side, str(output_path))
        print(f"[SAVE] Saved to: {output_path}")
        
        # Optionally add labels using PIL
        if add_labels:
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.open(output_path)
                draw = ImageDraw.Draw(img)
                
                # Try to use a larger font, fallback to default
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                # Add labels
                width = img.width
                draw.text((10, 10), "Input/Measurement", fill='white', font=font)
                draw.text((width // 2 + 10, 10), "Reconstructed", fill='white', font=font)
                
                img.save(output_path)
            except Exception as e:
                print(f"[WARNING] Could not add labels: {e}")
                
    elif reconstructed is not None:
        save_image(reconstructed, str(output_path))
        print(f"[SAVE] Saved to: {output_path}")
    else:
        print(f"[WARNING] No images to save")


def save_trajectory(trajectory, output_dir, image_idx, image_name):
    """
    Save intermediate sampling steps
    
    Args:
        trajectory: List of dicts with 'x0hat', 'x0y', 'xt' in [-1, 1] range
        output_dir: Base output directory
        image_idx: Image index
        image_name: Base name for the image
    """
    traj_dir = Path(output_dir) / 'intermediate' / f'{image_idx:04d}_{image_name}'
    traj_dir.mkdir(parents=True, exist_ok=True)
    
    # Save key steps (e.g., every 10th step + first and last)
    save_steps = list(range(0, len(trajectory), max(1, len(trajectory) // 10)))
    if len(trajectory) - 1 not in save_steps:
        save_steps.append(len(trajectory) - 1)
    
    for i in save_steps:
        step_data = trajectory[i]
        step_num = step_data['step']
        sigma = step_data['sigma']
        
        # Convert from [-1, 1] to [0, 1] for saving
        # save_image will convert [0, 1] -> [0, 255]
        x0hat = (step_data['x0hat'] * 0.5 + 0.5).clamp(0, 1)
        save_image(x0hat, str(traj_dir / f'step_{step_num:04d}_sigma_{sigma:.4f}_x0hat.png'))
        
        x0y = (step_data['x0y'] * 0.5 + 0.5).clamp(0, 1)
        save_image(x0y, str(traj_dir / f'step_{step_num:04d}_sigma_{sigma:.4f}_x0y.png'))
    
    print(f"[SAVE] Saved {len(save_steps)} intermediate steps to: {traj_dir.name}/")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# Main Sampling Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Posterior sampling for microscopy diffusion models')
    parser.add_argument('--config', type=str, 
                       default='./config/posterior_sampling.yaml',
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
    parser.add_argument('--random_select', action='store_true',
                       help='Randomly select images instead of taking first n')
    
    args = parser.parse_args()
    
    # Load config file
    print("=" * 80)
    print("Posterior Sampling for Microscopy Diffusion")
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
    output_dir = config['output']['save_dir']
    num_samples = config['data']['num_samples']
    random_select = config['data'].get('random_select', False)
    images_are_clean = config['data'].get('images_are_clean', True)
    save_intermediate = config['output'].get('save_intermediate', False)
    add_labels = config['output'].get('add_labels', False)
    device = config['device']['device']
    
    # DAPS parameters
    daps_config = config['daps']
    annealing_steps = daps_config['annealing_steps']
    sigma_max = daps_config['sigma_max']
    sigma_min = daps_config['sigma_min']
    mcmc_steps = daps_config['mcmc_steps']
    mcmc_lr = daps_config['mcmc_lr']
    tau = daps_config['tau']
    solver = daps_config.get('solver', 'tweedie')  # NEW: ODE solver type
    num_ode_steps = daps_config.get('num_ode_steps', 50)  # NEW: ODE integration steps
    
    # Operator parameters
    operator_config = config['operator']
    kernel_size = operator_config['kernel_size']
    blur_intensity = operator_config['intensity']
    noise_sigma = operator_config['noise_sigma']
    
    # Advanced options
    adv_config = config['advanced']
    seed = adv_config['seed']
    verbose = adv_config['verbose']
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n[CONFIG] Parameters:")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Test dir: {test_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num samples: {num_samples}")
    print(f"  Device: {device}")
    print(f"  Annealing steps: {annealing_steps}")
    print(f"  MCMC steps: {mcmc_steps}")
    print(f"  Tau: {tau}")
    print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
    print(f"  ODE Solver: {solver}" + (f" ({num_ode_steps} steps)" if solver != 'tweedie' else " (direct denoising)"))
    
    # Load model
    model = MicroscopyDiffusionModel(checkpoint, device=device)
    
    # Create forward operator (Gaussian blur)
    operator = GaussianBlur(
        kernel_size=kernel_size,
        intensity=blur_intensity,
        device=device,
        sigma=noise_sigma
    )
    print(f"[OPERATOR] Gaussian blur: kernel_size={kernel_size}, intensity={blur_intensity}, noise_sigma={noise_sigma}")
    
    # Create DAPS sampler
    sampler = DAPS(
        annealing_steps=annealing_steps,
        mcmc_steps=mcmc_steps,
        mcmc_lr=mcmc_lr,
        tau=tau,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        solver=solver,
        num_ode_steps=num_ode_steps
    )
    
    # Get test images
    test_dir = Path(test_dir)
    all_image_files = sorted(list(test_dir.glob('*.png')))
    
    if not all_image_files:
        print(f"[ERROR] No images found in {test_dir}")
        return
    
    print(f"[DATA] Found {len(all_image_files)} total images in test directory")
    
    # Select images - either random or first n
    if random_select:
        if num_samples > len(all_image_files):
            print(f"[WARNING] Requested {num_samples} samples but only {len(all_image_files)} available")
            num_samples = len(all_image_files)
        
        # Random selection without replacement
        import random
        random.seed(seed)
        image_files = random.sample(all_image_files, num_samples)
        print(f"[DATA] Randomly selected {len(image_files)} images")
    else:
        image_files = all_image_files[:num_samples]
        print(f"[DATA] Using first {len(image_files)} images")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_path.name}")
        
        # Load image (normalized to [-1, 1] for model)
        loaded_image = load_image(image_path).to(device)
        
        # Handle measurement creation based on whether images are clean or already degraded
        if images_are_clean:
            # Images are clean - apply operator to create degraded measurements
            with torch.no_grad():
                measurement = operator.measure(loaded_image)
                # Clip to ensure measurement stays in model's expected range [-1, 1]
                measurement = torch.clamp(measurement, -1.0, 1.0)
            input_for_display = measurement  # Show degraded measurement on left
        else:
            # Images are already degraded - use directly as measurements
            measurement = torch.clamp(loaded_image, -1.0, 1.0)
            input_for_display = loaded_image  # Show degraded input on left
        
        # Get initial sample from prior
        x_start = sampler.get_start(1, model)
        
        # Perform posterior sampling
        print(f"[SAMPLING] Running DAPS...")
        with torch.no_grad():
            if save_intermediate:
                reconstructed, trajectory = sampler.sample(
                    model, x_start, operator, measurement, verbose=verbose, save_trajectory=True
                )
                # Save trajectory
                save_trajectory(trajectory, output_dir, idx, image_path.stem)
            else:
                reconstructed = sampler.sample(
                    model, x_start, operator, measurement, verbose=verbose, save_trajectory=False
                )
        
        # Save final results: input (degraded) | reconstructed (clean)
        # Both are in [-1, 1], save_results will convert to [0, 1] -> [0, 255]
        output_path = output_dir / f'result_{idx:04d}_{image_path.stem}.png'
        save_results({
            'measurement': input_for_display[0],  # Degraded input in [-1, 1]
            'reconstructed': reconstructed[0]      # Reconstructed output in [-1, 1]
        }, output_path, add_labels=add_labels)
    
    print("\n" + "=" * 80)
    print(f"[COMPLETE] All samples saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

