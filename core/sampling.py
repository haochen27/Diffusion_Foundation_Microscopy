"""
Centralized Sampling System for Microscopy Diffusion Models
==========================================================
Extracts and unifies sampling logic from EDM scheduler and Gaussian diffusion.
Supports DDPM, DDIM, and EDM Euler sampling with full conditioning support.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Local imports
from .models import MicroscopyDiTModel
from .unet.model import MicroscopyUnetModel
from .edm_scheduler import EDMEulerScheduler, EDMPreconditioner
from .conditioning import DiTConditionEncoder, DiTConditionInjector, create_microscopy_conditions
from diffusers.models import AutoencoderKL


class MicroscopySampler:
    """
    Centralized sampling system for microscopy diffusion models.
    Supports multiple sampling methods with full conditioning capabilities.
    """
    
    def __init__(self, config_path: str):
        """Initialize sampler with configuration"""
        # initialization log
        self.config = self._load_config(config_path)
        
        self.device = self._setup_device()
        self.model = None
        self.vae = None
        self.diffusion = None
        self.scheduler = None
        self.condition_encoder = None
        self.condition_injector = None

        # Setup components
        
        self._setup_model()
        
        self._setup_vae()
        
        self._setup_sampling()
        
        self._setup_conditioning()
        
        
    def _load_config(self, config_path: str) -> Dict:
        """Load sampling configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        device_config = self.config.get('device', {})
        device_str = device_config.get('device', 'auto')
        
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)
        
        
        return device
    
    def _setup_model(self):
        """Load trained model from checkpoint"""
        checkpoint_path = self.config['model']['checkpoint_path']
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        

        # Load checkpoint with mmap for large files
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True)
            
        except Exception as e:
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model configuration from checkpoint
        if 'hyper_parameters' in checkpoint:
            model_config = checkpoint['hyper_parameters']['config']
            phase_config = checkpoint['hyper_parameters']['phase_config']
        else:
            # Fallback: use default configs
            model_config = self._get_default_model_config()
            phase_config = self._get_default_phase_config()
        
        # Create model based on architecture
        arch = str(model_config.get('model', {}).get('architecture', model_config.get('architecture', 'DiT-S/8')))
        if arch.lower() == 'unet':
            self.model = MicroscopyUnetModel(model_config, phase_config)
        else:
            self.model = MicroscopyDiTModel(model_config, phase_config)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        
    
    def _setup_vae(self):
        """Setup VAE for latent space decoding"""
        vae_config = self.config.get('vae', {})
        
        if not vae_config.get('enabled', True):
            
            self.vae = None
            return
        
        try:
            model_id = vae_config.get('model_id', 'stabilityai/stable-diffusion-2-1')
            self.vae = AutoencoderKL.from_pretrained(model_id)
            self.vae.requires_grad_(False)
            self.vae.to(self.device)
            self.vae.eval()
            
        except Exception as e:
            
            
            self.vae = None
    
    def _setup_sampling(self):
        """Setup sampling method based on configuration"""
        # Only EDM sampling is supported
        self._setup_edm_sampling()
    
    def _setup_edm_sampling(self):
        """Setup EDM Euler sampling"""

        # Always create our own EDM scheduler with correct prediction type
        # Don't reuse model scheduler as it might have wrong prediction_type from checkpoint
        
        
        # Prefer training-style scheduler block if present to align with training
        scheduler_config = self.config.get('scheduler', {})
        sampling_config = self.config.get('sampling', {})
        edm_config = sampling_config.get('edm', {})

        # Resolve parameters with priority: scheduler block -> sampling.edm/model -> defaults
        num_train_timesteps = scheduler_config.get('num_train_timesteps')
        if num_train_timesteps is None:
            # Fallback to model-specified timesteps as in previous code
            model_config = self.config.get('model', {})
            num_train_timesteps = model_config.get('num_train_timesteps', 1000)

        sigma_min = scheduler_config.get('sigma_min', edm_config.get('sigma_min', 0.002))
        sigma_max = scheduler_config.get('sigma_max', edm_config.get('sigma_max', 80.0))
        rho = scheduler_config.get('rho', edm_config.get('rho', 7.0))
        sigma_data = scheduler_config.get('sigma_data', 0.4)
        prediction_type = scheduler_config.get('prediction_type', 'sample')

        # Create EDM scheduler aligned with training
        self.scheduler = EDMEulerScheduler(
            num_train_timesteps=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
            prediction_type=prediction_type
        )

        # Setup preconditioner if conditional
        if self.config['model']['phase'] == 'conditional':
            self.edm_preconditioner = EDMPreconditioner(
                model=self.model.model,
                sigma_data=sigma_data,
                prediction_type=prediction_type
            )

        self.use_edm = True
    
    def _setup_conditioning(self):
        """Setup conditioning for Phase 2 models"""
        if self.config['model']['phase'] != 'conditional':
            return
        
        conditioning_config = self.config.get('conditioning', {})
        if not conditioning_config.get('enabled', True):
            return
        
        condition_types = conditioning_config.get('condition_types', [])
        
        if condition_types:
            # Setup condition encoder
            self.condition_encoder = DiTConditionEncoder(
                condition_types=condition_types,
                hidden_size=768  # DiT-XL hidden size
            )
            
            # Setup condition injector
            self.condition_injector = DiTConditionInjector(hidden_size=768)
            
            
    
    def generate_samples(
        self,
        num_samples: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
        condition_hints: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Generate samples using the configured sampling method.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Output image size (H, W)
            seed: Random seed for reproducibility
            condition_hints: Optional condition hints for conditional sampling
            
        Returns:
            Generated samples as tensor
        """
        # Use config defaults if not provided
        if num_samples is None:
            num_samples = self.config['output']['num_samples']
        if image_size is None:
            image_size = tuple(self.config['output']['image_size'])
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        
        
        # Determine latent shape
        if self.vae is not None:
            # VAE latent space: typically 4x64x64 for SD VAEs
            latent_size = image_size[0] // 8  # VAE downsampling factor
            shape = (num_samples, 4, latent_size, latent_size)
        else:
            # Pixel space: single channel microscopy
            shape = (num_samples, 1, image_size[0], image_size[1])
        
        # Generate samples
        samples = self._generate_edm_samples(shape, condition_hints)
        
        # Decode from latent space if using VAE
        if self.vae is not None:
            samples = self._decode_vae_samples(samples)
        
        return samples
    
    def _denoise(self, x, sigma_vec, shape, condition_hints, use_cuda, amp_dtype, y_label):
        """Compute denoised estimate x0_hat at given sigma. Shared by Euler & Heun."""
        bsz = shape[0]
        if hasattr(self, 'edm_preconditioner') and self.edm_preconditioner is not None:
            if use_cuda:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    return self.edm_preconditioner(x, sigma_vec)
            else:
                return self.edm_preconditioner(x, sigma_vec)
        else:
            sigma_data = self.scheduler.sigma_data
            sigma_i = sigma_vec[0]  # scalar
            c_in = 1.0 / torch.sqrt(sigma_data**2 + sigma_i**2)
            c_noise_val = sigma_i.log() / 4.0
            x_in = c_in.view(-1, 1, 1, 1) * x
            t_vec = c_noise_val.expand(bsz)

            if condition_hints is not None and hasattr(self, 'condition_encoder'):
                condition_emb = self._encode_conditions(condition_hints)
                if use_cuda:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        model_output = self.model.forward(x_in, t_vec, condition_emb)
                else:
                    model_output = self.model.forward(x_in, t_vec, condition_emb)
            else:
                if use_cuda:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        model_output = self.model.model(x_in, t_vec, y_label)
                else:
                    model_output = self.model.model(x_in, t_vec, y_label)

            in_channels = x.shape[1]
            if model_output.shape[1] != in_channels:
                model_output = model_output[:, :in_channels, ...]
            model_output = model_output.to(dtype=x.dtype)

            c_skip = sigma_data**2 / (sigma_i**2 + sigma_data**2)
            c_out = sigma_i * sigma_data / torch.sqrt(sigma_i**2 + sigma_data**2)
            return c_skip.view(-1,1,1,1) * x + c_out.view(-1,1,1,1) * model_output

    def _generate_edm_samples(self, shape: Tuple[int, ...], condition_hints: Optional[Dict] = None) -> torch.Tensor:
        """Generate samples using EDM Heun sampling (Algorithm 2, Karras et al. 2022)."""
        sampling_config = self.config['sampling']
        num_steps = int(sampling_config['num_steps'])
        use_heun = sampling_config.get('solver', 'heun') == 'heun'

        # Setup intermediate saving
        save_intermediate = self.config['advanced'].get('save_intermediate', False)
        intermediate_dir = None
        if save_intermediate:
            intermediate_dir = Path(self.config['advanced'].get('intermediate_dir', 'intermediate_samples'))
            intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Get sigma schedule (decreasing from sigma_max -> sigma_min)
        sigmas = self.scheduler.sigmas.to(self.device)

        # Optionally sub-sample to num_steps
        if num_steps is not None and num_steps < len(sigmas):
            idxs = torch.linspace(0, len(sigmas)-1, steps=num_steps).round().long()
            sigmas = sigmas[idxs]

        # Append t_N = 0 (EDM convention: generate.py line 33)
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

        # Initialize x ~ N(0, sigma_max^2 I)
        x = torch.randn(shape, device=self.device) * sigmas[0]

        if save_intermediate:
            self._save_intermediate_step(x, intermediate_dir, 0, "noise")

        num_solver_steps = len(sigmas) - 1
        if self.config['advanced'].get('show_progress', True):
            step_iter = tqdm(range(num_solver_steps), desc=f"EDM {'Heun' if use_heun else 'Euler'}")
        else:
            step_iter = range(num_solver_steps)

        # Mixed precision toggle for memory headroom
        use_cuda = torch.cuda.is_available() and self.device.type == 'cuda'
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        y_label = torch.zeros(shape[0], dtype=torch.long, device=self.device)
        bsz = shape[0]

        with torch.no_grad():
            for i in step_iter:
                sigma_i = sigmas[i]
                sigma_ip1 = sigmas[i+1]

                # Denoised estimate at current sigma
                x0_hat = self._denoise(x, sigma_i.expand(bsz), shape, condition_hints,
                                       use_cuda, amp_dtype, y_label)

                # Euler step: d = (x - x0_hat) / sigma
                d_cur = (x - x0_hat) / sigma_i.view(-1,1,1,1)
                x_next = x + (sigma_ip1 - sigma_i).view(-1,1,1,1) * d_cur

                # Heun's 2nd order correction (EDM Algorithm 2)
                if use_heun and i < num_solver_steps - 1 and sigma_ip1 > 0:
                    x0_hat_next = self._denoise(x_next, sigma_ip1.expand(bsz), shape,
                                                condition_hints, use_cuda, amp_dtype, y_label)
                    d_prime = (x_next - x0_hat_next) / sigma_ip1.view(-1,1,1,1)
                    x_next = x + (sigma_ip1 - sigma_i).view(-1,1,1,1) * (0.5 * d_cur + 0.5 * d_prime)

                x = x_next

                # Save intermediate steps
                if save_intermediate and intermediate_dir is not None:
                    if (i + 1) % 10 == 0 or (i + 1) == num_solver_steps:
                        step_name = f"step_{i+1:03d}_sigma_{sigma_ip1.item():.3f}"
                        self._save_intermediate_step(x, intermediate_dir, i + 1, step_name)

        # Save final result if intermediate saving is enabled
        if save_intermediate and intermediate_dir is not None:
            self._save_intermediate_step(x, intermediate_dir, len(sigmas), "final")

        return x
    
    def _encode_conditions(self, condition_hints: Dict) -> torch.Tensor:
        """Encode condition hints for conditional sampling"""
        if not hasattr(self, 'condition_encoder'):
            return None
        
        # Create condition dictionary
        conditions = {}
        condition_types = self.config['conditioning']['condition_types']
        
        for i, condition_type in enumerate(condition_types):
            if condition_type in condition_hints:
                conditions[condition_type] = condition_hints[condition_type]
        
        if conditions:
            return self.condition_encoder(conditions)
        
        return None
    
    def _decode_vae_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Decode samples from VAE latent space to pixel space"""
        if self.vae is None:
            return samples
        
        # Decode latents
        with torch.no_grad():
            decoded = self.vae.decode(samples / self.config['vae']['scaling_factor']).sample

        # VAE typically outputs [-1, 1], keep as is since we handle conversion in save_samples
        # decoded will be converted from [-1,1] to [0,1] in save_samples function
        
        # Convert to grayscale (take first channel)
        if decoded.shape[1] == 3:
            decoded = decoded[:, :1, :, :]
        
        return decoded
    
    def save_samples(self, samples: torch.Tensor, save_dir: str, prefix: str = "sample") -> List[str]:
        """Save generated samples to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # Convert to numpy
        samples_np = samples.detach().cpu().numpy()

        # Convert from [-1,1] to [0,1] for saving as images
        samples_np = (samples_np + 1.0) / 2.0

        # Ensure values are in [0,1] range
        samples_np = np.clip(samples_np, 0, 1)
        
        # Save individual samples
        if self.config['output'].get('save_individual', True):
            for i, sample in enumerate(samples_np):
                # Convert to PIL Image
                if sample.shape[0] == 1:  # Grayscale
                    img_array = (sample[0] * 255).astype(np.uint8)
                    
                    img = Image.fromarray(img_array, mode='L')
                else:  # RGB
                    img_array = (sample.transpose(1, 2, 0) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='RGB')
                
                # Save
                filename = f"{prefix}_{i:03d}.{self.config['output']['image_format']}"
                save_path = save_dir / filename
                img.save(save_path, dpi=(self.config['output']['dpi'], self.config['output']['dpi']))
                saved_paths.append(str(save_path))
        
        # Save sample grid
        if self.config['output'].get('save_grid', True):
            grid_path = self._save_sample_grid(samples_np, save_dir, prefix)
            if grid_path:
                saved_paths.append(grid_path)
        
        return saved_paths

    def _save_intermediate_step(self, samples: torch.Tensor, intermediate_dir: Path, step: int, step_name: str):
        """Save intermediate sampling step"""
        try:
            # Convert to numpy and normalize
            samples_np = samples.detach().cpu().numpy()
            samples_np = (samples_np + 1.0) / 2.0
            samples_np = np.clip(samples_np, 0, 1)

            # Save first sample only for intermediate steps
            if samples_np.shape[0] > 0:
                sample = samples_np[0]  # First sample
                if sample.shape[0] == 1:  # Grayscale
                    img_array = (sample[0] * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                else:  # RGB
                    img_array = (sample.transpose(1, 2, 0) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='RGB')

                # Save with descriptive filename
                filename = f"{step_name}.png"
                save_path = intermediate_dir / filename
                img.save(save_path, dpi=(150, 150))

        except Exception as e:
            print(f"[WARNING] Failed to save intermediate step {step}: {e}")

    def _save_sample_grid(self, samples: np.ndarray, save_dir: Path, prefix: str) -> Optional[str]:
        """Save sample grid as single image"""
        try:
            num_samples = len(samples)
            grid_layout = self.config['output'].get('grid_layout', 'auto')
            
            if grid_layout == 'auto':
                grid_size = int(np.ceil(np.sqrt(num_samples)))
            else:
                grid_size = int(grid_layout.split('x')[0])
            
            # Create figure
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            if grid_size == 1:
                axes = [axes]
            elif grid_size > 1:
                axes = axes.flatten()
            
            # Plot samples
            for i in range(num_samples):
                if i < len(axes):
                    ax = axes[i]
                    if samples[i].shape[0] == 1:  # Grayscale
                        ax.imshow(samples[i][0], cmap='gray')
                    else:  # RGB
                        ax.imshow(samples[i].transpose(1, 2, 0))
                    ax.set_title(f'Sample {i+1}')
                    ax.axis('off')
            
            # Hide empty subplots
            for i in range(num_samples, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Save
            grid_filename = f"{prefix}_grid.{self.config['output']['image_format']}"
            grid_path = save_dir / grid_filename
            plt.savefig(grid_path, dpi=self.config['output']['dpi'], bbox_inches='tight')
            plt.close()
            
            return str(grid_path)
        
        except Exception as e:
            
            return None
    
    
    
    def _get_default_model_config(self) -> Dict:
        """Get default model configuration"""
        return {
            'model': {
                'architecture': 'DiT-XL/8',
                'input_size': [512, 512],
                'hidden_size': 768,
                'latent_size': 64,
                'latent_channels': 4
            },
            'vae': {'enabled': True, 'model_id': 'stabilityai/stable-diffusion-2-1'},
            'optimizer': {'lr': 1e-4, 'weight_decay': 0.01}
        }
    
    def _get_default_phase_config(self) -> Dict:
        """Get default phase configuration"""
        return {
            'type': 'unconditional',
            'condition_types': [],
            'condition_dropout': 0.0
        }
    
    @classmethod
    def _create_from_module(cls, pl_module, config: Dict):
        """Create sampler instance from existing Lightning module (for callbacks)"""
        sampler = cls.__new__(cls)
        sampler.config = config
        sampler.device = next(pl_module.parameters()).device
        sampler.model = pl_module
        sampler.vae = getattr(pl_module, 'vae', None)
        sampler.diffusion = getattr(pl_module, 'diffusion', None)
        sampler.scheduler = getattr(pl_module, 'scheduler', None)
        sampler.use_edm = getattr(pl_module, 'use_edm', False)
        sampler.condition_encoder = getattr(pl_module, 'condition_encoder', None)
        sampler.condition_injector = getattr(pl_module, 'condition_injector', None)
        sampler.edm_preconditioner = getattr(pl_module, 'edm_preconditioner', None)
        return sampler


def create_condition_hints(
    image_size: Tuple[int, int],
    condition_types: List[str],
    hints_config: Dict
) -> Dict:
    """
    Create condition hints for conditional sampling.
    
    Args:
        image_size: Target image size (H, W)
        condition_types: List of condition types to generate
        hints_config: Configuration for hint generation
        
    Returns:
        Dictionary of condition hints
    """
    hints = {}
    
    for condition_type in condition_types:
        if condition_type == 'super_resolution':
            # Create low-resolution version
            scale = hints_config.get('super_resolution_scale', 2.0)
            low_res_size = (int(image_size[0] / scale), int(image_size[1] / scale))
            hints[condition_type] = torch.randn(1, 1, *low_res_size)
            
        elif condition_type == 'denoising':
            # Create noisy version
            hints[condition_type] = torch.randn(1, 1, *image_size) * hints_config.get('denoising_strength', 0.5)
            
        elif condition_type == 'canny_edge':
            # Create random edge map
            hints[condition_type] = torch.randn(1, 1, *image_size)
            
        elif condition_type == 'depth':
            # Create random depth map
            hints[condition_type] = torch.randn(1, 1, *image_size)
            
        else:
            # Generic condition
            hints[condition_type] = torch.randn(1, 1, *image_size)
    
    return hints