"""
EDM (Elucidating the Design Space of Diffusion Models) Scheduler
===============================================================
Implementation of the EDM scheduler from Karras et al. 2022
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Tuple
import math


class EDMEulerScheduler:
    """
    EDM Euler scheduler for diffusion models
    
    Based on "Elucidating the Design Space of Diffusion Models" by Karras et al.
    Uses continuous-time formulation with better noise schedules.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.4,
        rho: float = 7.0,
        prediction_type: str = "sample",
        device: Optional[torch.device] = None
    ):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.prediction_type = prediction_type
        self.device = device
        
        # Generate sigma schedule
        self.sigmas = self._generate_sigma_schedule()
        
        print(f"[EDM] Initialized scheduler:")
        print(f"[EDM] - Timesteps: {num_train_timesteps}")
        print(f"[EDM] - Sigma range: [{sigma_min}, {sigma_max}]")
        print(f"[EDM] - Prediction type: {prediction_type}")
    
    def _generate_sigma_schedule(self) -> torch.Tensor:
        """Generate the sigma schedule according to EDM paper"""
        
        # EDM sigma schedule: σ(i) = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
        rho_inv = 1.0 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        
        timesteps = np.linspace(0, 1, self.num_train_timesteps)
        sigmas = (sigma_max_rho + timesteps * (sigma_min_rho - sigma_max_rho)) ** self.rho
        
        return torch.tensor(sigmas, dtype=torch.float32)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples according to EDM formulation"""
        
        # Get sigmas for the given timesteps (index on CPU, then move)
        sigmas = self.sigmas[timesteps.detach().to('cpu')].to(original_samples.device)
        
        # Reshape sigmas to match sample dimensions
        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)
        
        # EDM noise addition: x_noisy = x + σ * noise
        noisy_samples = original_samples + sigmas * noise
        
        return noisy_samples
    
    def get_scalings_for_boundary_condition(
        self,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get scaling factors for boundary condition"""
        
        sigmas = self.sigmas[timesteps.detach().to('cpu')].to(timesteps.device)
        
        # EDM preconditioning
        c_skip = self.sigma_data**2 / (sigmas**2 + self.sigma_data**2)
        c_out = sigmas * self.sigma_data / torch.sqrt(sigmas**2 + self.sigma_data**2)
        
        return c_skip, c_out
    
    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """Scale model input according to EDM preconditioning"""
        
        sigmas = self.sigmas[timestep.detach().to('cpu')].to(sample.device)
        
        # Reshape sigmas to match sample dimensions
        while len(sigmas.shape) < len(sample.shape):
            sigmas = sigmas.unsqueeze(-1)
        
        # EDM input scaling: c_in * x
        c_in = 1.0 / torch.sqrt(sigmas**2 + self.sigma_data**2)
        scaled_sample = c_in * sample
        
        return scaled_sample
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Get velocity parameterization target"""
        
        sigmas = self.sigmas[timesteps.detach().to('cpu')].to(sample.device)
        
        # Reshape sigmas to match sample dimensions
        while len(sigmas.shape) < len(sample.shape):
            sigmas = sigmas.unsqueeze(-1)
        
        # v-parameterization: v = α_t * ε - σ_t * x_0
        alpha_t = 1.0 / torch.sqrt(1.0 + sigmas**2 / self.sigma_data**2)
        sigma_t = sigmas / self.sigma_data / torch.sqrt(1.0 + sigmas**2 / self.sigma_data**2)
        
        velocity = alpha_t * noise - sigma_t * sample
        
        return velocity
    
    def get_loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get loss weights according to EDM paper"""
        
        sigmas = self.sigmas[timesteps.detach().to('cpu')].to(timesteps.device)
        
        # EDM loss weighting: λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
        weights = (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data)**2
        
        return weights
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Tuple]:
        """
        Predict the sample at the previous timestep by solving the probability flow ODE.
        
        This is the Euler method for solving the ODE.
        """
        
        if timestep >= len(self.sigmas):
            timestep = len(self.sigmas) - 1
        
        sigma = self.sigmas[timestep].to(sample.device)
        sigma_prev = self.sigmas[timestep - 1].to(sample.device) if timestep > 0 else torch.tensor(0.0).to(sample.device)
        
        # Reshape sigmas
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)
            sigma_prev = sigma_prev.unsqueeze(-1)
        
        # The model output from the preconditioner is the denoised sample (x0_hat)
        denoised = model_output
        
        # Euler step: x_{t-1} = x_t + (σ_{t-1} - σ_t) * d_x
        derivative = (sample - denoised) / sigma
        prev_sample = sample + (sigma_prev - sigma) * derivative
        
        if not return_dict:
            return (prev_sample,)
        
        return {"prev_sample": prev_sample}
    
    def add_noise_to_timesteps(
        self,
        timesteps: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Convert timesteps to sigma values"""
        
        if device is not None:
            timesteps = timesteps.to(device)
        
        return self.sigmas[timesteps]


class EDMPreconditioner(nn.Module):
    """
    EDM Preconditioning wrapper for diffusion models
    
    Implements the preconditioning scheme from EDM paper to improve training stability.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sigma_data: float = 0.4,
        prediction_type: str = "sample"
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.prediction_type = prediction_type
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Apply EDM preconditioning and call the underlying model
        """
        
        # Reshape sigma to match input dimensions
        while len(sigma.shape) < len(x.shape):
            sigma = sigma.unsqueeze(-1)
        
        # EDM preconditioning
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = sigma.log() / 4.0
        
        # Apply input scaling
        x_scaled = c_in * x
        
        # Convert sigma to timestep (approximate) as a per-sample scalar
        # This is a simplification - in practice, you'd want a proper sigma->timestep mapping
        timestep = c_noise.reshape(x.shape[0])  # shape [B]
        
        # Call underlying model with scaled input
        # Always provide a dummy label for DiT models that expect y
        batch_size = x.shape[0]
        dummy_y = None
        try:
            dummy_y = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        except Exception:
            dummy_y = None
        if dummy_y is not None:
            model_output = self.model(x_scaled, timestep, dummy_y, **model_kwargs)
        else:
            model_output = self.model(x_scaled, timestep, **model_kwargs)
        
        # Ensure channel compatibility: some models output 2*C (mean,var). Keep first C.
        in_channels = x.shape[1]
        if model_output.shape[1] != in_channels:
            model_output = model_output[:, :in_channels, ...]

        # Apply output scaling based on prediction type
        if self.prediction_type == "sample":
            # F_theta is the model output, which is the denoised sample
            result = c_skip * x + c_out * model_output
        elif self.prediction_type == "epsilon":
            # model predicts epsilon, convert to denoised sample x0_hat
            result = x - sigma * model_output
        else:
            # Fallback to sample prediction (original EDM)
            result = c_skip * x + c_out * model_output
        
        return result


def create_edm_scheduler(config: dict) -> EDMEulerScheduler:
    """Create EDM scheduler from config"""
    
    scheduler_config = config.get('scheduler', {})
    
    return EDMEulerScheduler(
        num_train_timesteps=scheduler_config.get('num_train_timesteps', 1000),
        sigma_min=scheduler_config.get('sigma_min', 0.002),
        sigma_max=scheduler_config.get('sigma_max', 80.0),
        sigma_data=scheduler_config.get('sigma_data', 0.4),
        rho=scheduler_config.get('rho', 7.0),
        prediction_type=scheduler_config.get('prediction_type', 'sample')
    )