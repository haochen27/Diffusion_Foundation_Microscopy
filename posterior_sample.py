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
import tqdm
from typing import Dict, Optional

# Import core modules
from core.unet.model import MicroscopyUnetModel
from core.edm_scheduler import EDMEulerScheduler


# ============================================================================
# Forward Operators (from DAPS structure)
# ============================================================================

class Operator:
    """Base operator class for measurement process"""
    
    def __init__(self, sigma=0.05):
        """
        Args:
            sigma: Measurement noise standard deviation
        """
        self.sigma = sigma
    
    def __call__(self, x):
        """Apply operator to input"""
        raise NotImplementedError
    
    def measure(self, x):
        """Measure input by applying operator and adding noise"""
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)
    
    def loss(self, x, y):
        """Compute squared-error loss"""
        return ((self(x) - y) ** 2).flatten(1).sum(-1)
    
    def gradient(self, x, y, return_loss=False):
        """Compute gradient of loss w.r.t input"""
        with torch.enable_grad():
            x_tmp = x.clone().detach().requires_grad_(True)
            loss = self.loss(x_tmp, y).sum()
            x_grad = torch.autograd.grad(loss, x_tmp)[0]
        if return_loss:
            return x_grad, loss
        return x_grad


class GaussianBlur(Operator):
    """Gaussian blur operator"""
    
    def __init__(self, kernel_size=31, intensity=3.0, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = self._create_blur_kernel(kernel_size, intensity).to(device)
        # Don't set requires_grad to False - we need gradients for MCMC
    
    def _create_blur_kernel(self, kernel_size, std):
        """Create Gaussian blur convolution layer"""
        import scipy.ndimage
        import torch.nn as nn
        
        # Create Gaussian kernel
        n = np.zeros((kernel_size, kernel_size))
        n[kernel_size // 2, kernel_size // 2] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=std)
        k = torch.from_numpy(k).float()
        
        # Create conv layer (for single channel)
        conv = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(1, 1, kernel_size, stride=1, padding=0, bias=False, groups=1)
        )
        
        # Set weights and freeze them (but allow gradients to flow through operations)
        conv[1].weight.data = k.unsqueeze(0).unsqueeze(0)
        for param in conv.parameters():
            param.requires_grad = False
        
        return conv
    
    def __call__(self, data):
        """Apply Gaussian blur"""
        return self.conv(data)


# ============================================================================
# Model Wrapper for DAPS
# ============================================================================

class MicroscopyDiffusionModel:
    """Wrapper for microscopy UNet model to work with DAPS sampler"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint and model
        print(f"[MODEL] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract configs
        if 'hyper_parameters' in checkpoint:
            self.config = checkpoint['hyper_parameters']['config']
            self.phase_config = checkpoint['hyper_parameters']['phase_config']
        else:
            raise ValueError("Checkpoint missing hyper_parameters")
        
        # Create model
        self.lightning_model = MicroscopyUnetModel(self.config, self.phase_config)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            self.lightning_model.load_state_dict(checkpoint['state_dict'])
            print(f"[MODEL] Loaded state dict")
        
        self.lightning_model.to(device)
        self.lightning_model.eval()
        
        # Access the EDM preconditioner and scheduler
        self.model = self.lightning_model.model
        self.edm_preconditioner = self.lightning_model.edm_preconditioner
        self.scheduler = self.lightning_model.scheduler
        
        print(f"[MODEL] Model loaded successfully")
        print(f"[MODEL] Scheduler sigma range: [{self.scheduler.sigma_min}, {self.scheduler.sigma_max}]")
    
    def score(self, x, sigma):
        """Compute score function: ∇log p(x_t)"""
        # Use Tweedie's formula: score = (D_θ(x_t, σ) - x_t) / σ^2
        d = self.tweedie(x, sigma)
        return (d - x) / (sigma ** 2)
    
    def tweedie(self, x, sigma):
        """Compute denoised estimate using EDM preconditioner"""
        with torch.no_grad():
            # Ensure sigma is tensor with correct shape
            if isinstance(sigma, (int, float)):
                sigma = torch.tensor([sigma] * x.shape[0], device=x.device)
            elif sigma.dim() == 0:
                sigma = sigma.unsqueeze(0).repeat(x.shape[0])
            
            # Ensure sigma is on correct device
            sigma = sigma.to(x.device)
            
            # Get denoised estimate from EDM preconditioner
            denoised = self.edm_preconditioner(x, sigma)
            
            return denoised
    
    def get_in_shape(self):
        """Return input shape for the model"""
        return (1, 512, 512)  # Single channel microscopy images


# ============================================================================
# MCMC Sampler (from DAPS cores/mcmc.py)
# ============================================================================

class MCMCSampler:
    """MCMC sampler for posterior sampling"""
    
    def __init__(self, num_steps=10, lr=0.1, tau=0.01, lr_min_ratio=0.01, 
                 prior_solver='gaussian', mc_algo='langevin', momentum=0.9):
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.mc_algo = mc_algo
        self.momentum = momentum
    
    def score_fn(self, x, x0hat, model, xt, operator, measurement, sigma):
        """Compute conditional score function"""
        # Data fitting term
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
        data_term = -data_fitting_grad / self.tau ** 2
        
        # Connection to noisy observation
        xt_term = (xt - x) / sigma ** 2
        
        # Prior term
        prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)
        
        # Combine terms with gradient clipping for stability
        total_score = data_term + xt_term + prior_term
        
        # Clip gradients to prevent NaN
        total_score = torch.clamp(total_score, -100.0, 100.0)
        
        return total_score, data_fitting_loss
    
    def get_prior_score(self, x, x0hat, xt, model, sigma):
        """Get prior score based on solver type"""
        if self.prior_solver == 'gaussian':
            return (x0hat - xt).detach() / sigma ** 2
        else:
            raise NotImplementedError(f"Prior solver {self.prior_solver} not implemented")
    
    def prepare_prior_score(self, x0hat, xt, model, sigma):
        """Precompute prior score"""
        if self.prior_solver == 'gaussian':
            self.prior_score = (x0hat - xt).detach() / sigma ** 2
    
    def mc_prepare(self, x0hat, xt, model, operator, measurement, sigma):
        """Prepare MCMC state"""
        if self.mc_algo == 'hmc':
            self.velocity = torch.randn_like(x0hat)
    
    def mc_update(self, x, cur_score, lr, epsilon):
        """Perform MCMC update step"""
        if self.mc_algo == 'langevin':
            x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
        elif self.mc_algo == 'hmc':
            step_size = np.sqrt(lr)
            self.velocity = self.momentum * self.velocity + step_size * cur_score + \
                          np.sqrt(2 * (1 - self.momentum)) * epsilon
            x_new = x + self.velocity * step_size
        else:
            raise NotImplementedError
        return x_new
    
    def get_lr(self, ratio):
        """Compute learning rate based on annealing ratio"""
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
    
    def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, verbose=False):
        """Perform MCMC sampling"""
        lr = self.get_lr(ratio)
        self.mc_prepare(x0hat, xt, model, operator, measurement, sigma)
        self.prepare_prior_score(x0hat, xt, model, sigma)
        
        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps, desc="MCMC steps") if verbose else range(self.num_steps)
        
        for _ in pbar:
            cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma)
            epsilon = torch.randn_like(x)
            x = self.mc_update(x, cur_score, lr, epsilon)
            
            # Clip values to prevent divergence
            x = torch.clamp(x, -10.0, 10.0)
            
            # Early stopping with NaN
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("[WARNING] NaN/Inf detected in MCMC sampling, returning x0hat")
                return x0hat.detach()
        
        return x.detach()


# ============================================================================
# Diffusion Scheduler (from DAPS cores/scheduler.py)
# ============================================================================

class EDMSchedulerForDAPS:
    """EDM-based scheduler for DAPS annealing"""
    
    def __init__(self, num_steps, sigma_max=10.0, sigma_min=0.002, rho=7.0):
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        
        # Generate sigma schedule
        rho_inv = 1.0 / rho
        timesteps = np.linspace(0, 1, num_steps)
        sigmas = (sigma_max ** rho_inv + timesteps * (sigma_min ** rho_inv - sigma_max ** rho_inv)) ** rho
        self.sigma_steps = torch.tensor(sigmas, dtype=torch.float32)
        
        print(f"[SCHEDULER] Created annealing schedule with {num_steps} steps")
        print(f"[SCHEDULER] Sigma range: [{sigma_min}, {sigma_max}]")
    
    def get_prior_sigma(self):
        """Get prior noise level"""
        return self.sigma_max


class DiffusionPFODE:
    """Probability Flow ODE for reverse diffusion"""
    
    def __init__(self, model, sigma_max):
        self.model = model
        self.sigma_max = sigma_max
    
    def sample(self, xt):
        """Sample from PF-ODE (simplified: just return denoised estimate)"""
        with torch.no_grad():
            # Use the current noise level to denoise
            sigma = torch.tensor([self.sigma_max], device=xt.device)
            x0hat = self.model.tweedie(xt, sigma)
        return x0hat


# ============================================================================
# DAPS Sampler (from DAPS sampler.py)
# ============================================================================

class DAPS:
    """Decoupled Annealing Posterior Sampling"""
    
    def __init__(self, annealing_steps=20, mcmc_steps=10, mcmc_lr=0.1, 
                 tau=0.01, sigma_max=10.0, sigma_min=0.002):
        self.annealing_scheduler = EDMSchedulerForDAPS(
            num_steps=annealing_steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min
        )
        self.mcmc_sampler = MCMCSampler(
            num_steps=mcmc_steps,
            lr=mcmc_lr,
            tau=tau
        )
        print(f"[DAPS] Initialized with {annealing_steps} annealing steps")
        print(f"[DAPS] MCMC: {mcmc_steps} steps, lr={mcmc_lr}, tau={tau}")
    
    def sample(self, model, x_start, operator, measurement, verbose=False, save_trajectory=False):
        """
        Perform DAPS posterior sampling
        
        Args:
            model: Diffusion model
            x_start: Initial noise sample
            operator: Forward operator (e.g., Gaussian blur)
            measurement: Observed measurement
            verbose: Show progress bar
            save_trajectory: Whether to save intermediate steps
        
        Returns:
            Final sampled image (or tuple of (final, trajectory) if save_trajectory=True)
        """
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1, desc="DAPS") if verbose else \
               range(self.annealing_scheduler.num_steps - 1)
        
        xt = x_start
        trajectory = [] if save_trajectory else None
        
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            
            # 1. Reverse diffusion (PF-ODE)
            with torch.no_grad():
                pfode_sampler = DiffusionPFODE(model, sigma)
                x0hat = pfode_sampler.sample(xt)
            
            # 2. MCMC update for data consistency
            ratio = step / (self.annealing_scheduler.num_steps - 1)
            x0y = self.mcmc_sampler.sample(
                xt, model, x0hat, operator, measurement, sigma, ratio, verbose=False
            )
            
            # Save intermediate results
            if save_trajectory:
                trajectory.append({
                    'step': step,
                    'sigma': sigma.item() if torch.is_tensor(sigma) else sigma,
                    'x0hat': x0hat.detach().cpu(),
                    'x0y': x0y.detach().cpu(),
                    'xt': xt.detach().cpu()
                })
            
            # 3. Forward diffusion to next noise level
            if step != self.annealing_scheduler.num_steps - 2:
                sigma_next = self.annealing_scheduler.sigma_steps[step + 1]
                xt = x0y + torch.randn_like(x0y) * sigma_next
            else:
                xt = x0y
            
            if verbose and isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix({'sigma': f'{sigma:.4f}'})
        
        if save_trajectory:
            return xt, trajectory
        return xt
    
    def get_start(self, batch_size, model):
        """Generate initial random sample from prior"""
        device = model.device
        in_shape = model.get_in_shape()
        prior_sigma = self.annealing_scheduler.get_prior_sigma()
        x_start = torch.randn(batch_size, *in_shape, device=device) * prior_sigma
        return x_start


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
        sigma_min=sigma_min
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

