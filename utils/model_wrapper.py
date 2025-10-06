"""
Model Wrapper for Posterior Sampling
=====================================
Wrapper for microscopy UNet model to work with DAPS sampler
"""

import torch
from core.unet.model import MicroscopyUnetModel


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
