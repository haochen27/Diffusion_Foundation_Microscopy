"""
Microscopy Diffusion Models
===========================
DiT-XL/8 based models for microscopy applications
"""

from re import T
import torch
import torch.nn as nn
import lightning as L
from typing import Dict, List, Optional
import os
from copy import deepcopy

# Local imports
from .dit_models import DiT_models
from .diffusion import create_diffusion
from .conditioning import DiTConditionEncoder, DiTConditionInjector, Condition, create_microscopy_conditions
from .edm_scheduler import EDMEulerScheduler, EDMPreconditioner, create_edm_scheduler
from diffusers.models import AutoencoderKL


class MicroscopyDiTModel(L.LightningModule):
    """DiT-XL/8 based model for microscopy generation"""
    
    def __init__(self, config: Dict, phase_config: Dict):
        super().__init__()
        self.config = config
        self.phase_config = phase_config
        self.save_hyperparameters()
        
        # Setup model architecture
        self.setup_model()
        self.setup_diffusion()
        self.setup_vae()
        
        # Training state
        self.automatic_optimization = True
        
    def setup_model(self):
        """Setup DiT model based on phase"""
        
        model_config = self.config['model']
        
        # Create DiT model. Use dummy single-class labels for unconditional setup.
        if self.config['vae']['enabled']:
            # VAE latent-space training: typically 4x64x64 for SD VAEs
            input_size = model_config.get('latent_size', 64)
            in_channels = model_config.get('latent_channels', 4)
        else:
            # Pixel-space training on microscopy images (single-channel)
            input_size = model_config['input_size'][0]
            in_channels = 1
        
        self.model = DiT_models[model_config['architecture']](
            input_size=input_size,
            in_channels=in_channels,
            num_classes=1,  # use a single dummy class; we pass y=zeros
            learn_sigma=True  # Predict variance for efficiency   
        )
        
        # Setup conditioning for Phase 2
        if self.phase_config['type'] == 'conditional':
            self.setup_conditioning()
        
        # Load base checkpoint if specified
        if self.phase_config.get('base_checkpoint'):
            self.load_base_checkpoint()
    
    def setup_conditioning(self):
        """Setup OminiControl-style conditioning for Phase 2"""
        
        hidden_size = self.config['model']['hidden_size']
        condition_types = self.phase_config.get('condition_types', [])
        
        # Use DiT-compatible condition encoder
        self.condition_encoder = DiTConditionEncoder(
            condition_types=condition_types,
            hidden_size=hidden_size
        )
        
        # Condition injector for DiT blocks
        self.condition_injector = DiTConditionInjector(hidden_size=hidden_size)
    
    def setup_diffusion(self):
        """Setup diffusion process with EDM scheduler support"""
        
        # Check if EDM scheduler is configured
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'default')
        
        if scheduler_type == 'EDMEulerScheduler':
            print("[DIFFUSION] Using EDM Euler Scheduler")
            self.scheduler = create_edm_scheduler(self.config)
            self.use_edm = True
            
            # Always wrap model with EDM preconditioning to keep train/sample aligned
            self.edm_preconditioner = EDMPreconditioner(
                model=self.model,
                sigma_data=scheduler_config.get('sigma_data', 0.4),
                prediction_type=scheduler_config.get('prediction_type', 'sample')
            )

            # EDM log-normal training hyperparameters (Karras et al. 2022, Table 1)
            self.P_mean = float(scheduler_config.get('P_mean', -1.2))
            self.P_std = float(scheduler_config.get('P_std', 1.2))
            self.sigma_data = float(scheduler_config.get('sigma_data', 0.4))
            
        else:
            print("[DIFFUSION] Using default diffusion")
            self.diffusion = create_diffusion(timestep_respacing="")
            self.use_edm = False
    
    def setup_vae(self):
        """Setup VAE for latent encoding (optional for pixel-level training)"""
        vae_config = self.config['vae']
        
        if not vae_config.get('enabled', True):
            print("[VAE] Disabled - using pixel-level training")
            self.vae = None
            return
        
        try:
            self.vae = AutoencoderKL.from_pretrained(vae_config['model_id'])
            self.vae.requires_grad_(False)
            self.vae.eval()
            print(f"[VAE] Loaded: {vae_config['model_id']}")
        except Exception as e:
            print(f"[VAE] Failed to load pretrained VAE '{vae_config.get('model_id')}': {e}")
            print("[VAE] Using lightweight dummy VAE encoder for fallback (suitable for smoke tests only)")
            import torch.nn.functional as F
            class _DummyDist:
                def __init__(self, z): self._z = z
                def sample(self): return self._z
            class _DummyEnc:
                def __init__(self, z): self.latent_dist = _DummyDist(z)
            class _DummyVAE:
                def encode(self_inner, images):
                    # Expect [B,C,H,W]; produce [B,4,64,64] latents
                    imgs = images
                    if imgs.dim() == 3:
                        imgs = imgs.unsqueeze(0)
                    if imgs.shape[1] == 1:
                        imgs = imgs.repeat(1, 3, 1, 1)
                    z = F.interpolate(imgs[:, :1, ...], size=(64, 64), mode='area').repeat(1, 4, 1, 1)
                    return _DummyEnc(z)
            self.vae = _DummyVAE()
    
    def load_base_checkpoint(self):
        """Load weights from Phase 1"""
        checkpoint_path = self.phase_config['base_checkpoint']
        
        if os.path.exists(checkpoint_path):
            print(f"[LOADING] Base weights from: {checkpoint_path}")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    
                    # Extract model weights
                    model_weights = {}
                    for key, value in state_dict.items():
                        if key.startswith('model.'):
                            new_key = key.replace('model.', '')
                            model_weights[new_key] = value
                    
                    # Load model weights
                    if model_weights:
                        self.model.load_state_dict(model_weights, strict=False)
                        print(f"[SUCCESS] Loaded {len(model_weights)} model parameters")
                
            except Exception as e:
                print(f"[ERROR] Error loading checkpoint: {e}")
    
    def encode_conditions(self, batch):
        """Encode conditioning inputs (OminiControl style)"""
        if self.phase_config['type'] != 'conditional':
            return None
        
        if not hasattr(self, 'condition_encoder'):
            return None
        
        # Prepare condition dictionary
        conditions = {}
        condition_types = self.phase_config.get('condition_types', [])
        
        for i, condition_type in enumerate(condition_types):
            condition_key = f'condition_{i}'
            if condition_key in batch:
                conditions[condition_type] = batch[condition_key]
        
        if conditions:
            return self.condition_encoder(conditions)
        
        return None
    
    def forward(self, x, t, condition_emb=None):
        """Forward pass with optional conditioning"""
        if condition_emb is not None and hasattr(self, 'condition_injector'):
            # Custom forward with condition injection
            return self._forward_with_conditioning(x, t, condition_emb)
        else:
            # Standard DiT forward
            # Always pass a dummy label tensor to DiT
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.model(x, t, y)
    
    def _forward_with_conditioning(self, x, t, condition_emb):
        """DiT forward pass with condition injection"""
        # Get DiT model components
        model = self.model
        
        # Standard DiT preprocessing
        x = model.x_embedder(x) + model.pos_embed
        t = model.t_embedder(t)
        
        # Apply conditioning to features
        if hasattr(self, 'condition_injector'):
            x = self.condition_injector(x, condition_emb)
        
        # DiT transformer blocks
        for block in model.blocks:
            x = block(x, t)
        
        # Final layer
        x = model.final_layer(x, t)
        x = model.unpatchify(x)
        
        return x
    
    def _compute_edm_loss(self, latents, condition_emb=None):
        """Compute loss using EDM log-normal σ sampling (Karras et al. 2022)"""
        
        batch_size = latents.shape[0]
        device = latents.device

        # EDM log-normal σ sampling (Karras et al. 2022, Eq. 5 / Table 1)
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()  # log σ ~ N(P_mean, P_std²)

        # Forward process: x_t = x_0 + σ · ε
        noise = torch.randn_like(latents)
        noisy_latents = latents + noise * sigma

        # Obtain per-sample sigma values for the preconditioner
        sigma_vals = sigma.reshape(batch_size).to(device)

        # Forward pass via EDM preconditioner (expects unscaled x_t)
        # This yields the denoised estimate x0_hat aligned with sampling logic
        if hasattr(self, 'edm_preconditioner'):
            model_output = self.edm_preconditioner(noisy_latents, sigma_vals)
        else:
            # Fallback (should not happen when EDM is enabled)
            y = torch.zeros(noisy_latents.shape[0], dtype=torch.long, device=noisy_latents.device)
            t_cont = sigma_vals.log() / 4.0
            model_output = self.model(noisy_latents, t_cont, y)

        # Compute target
        scheduler_config = self.config.get('scheduler', {})
        prediction_type = scheduler_config.get('prediction_type', 'sample')

        if prediction_type == "sample":
            # Compare denoised estimate x0_hat against clean latents x0
            target = latents
        elif prediction_type == "epsilon":
            target = noise
        else:
            # Default to sample prediction (original EDM)
            target = latents

        # Ensure shapes match for MSE (model_output and target must have same channels)
        if model_output.shape != target.shape:
            # If model predicts 2*C (mean, var) and target is C, use the mean part only
            if model_output.shape[1] == target.shape[1] * 2:
                model_output = model_output[:, :target.shape[1], ...]

        # EDM loss weighting: λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        loss = weight * (model_output - target) ** 2

        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        """Training step"""

        # Get images
        images = batch['image']

        # Prepare inputs based on VAE configuration
        if self.vae is not None:
            # VAE latent space training
            with torch.no_grad():
                # Convert grayscale to RGB for VAE
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)

                # Encode to latent space
                latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        else:
            # Pixel-level training - use images directly
            latents = images
            # Ensure single channel for microscopy
            if latents.shape[1] == 3:
                latents = latents[:, :1, :, :]  # Take first channel only
        
        # Get conditioning
        condition_emb = self.encode_conditions(batch)
        
        # Apply condition dropout during training
        if self.training and condition_emb is not None:
            dropout_prob = self.phase_config.get('condition_dropout', 0.0)
            if torch.rand(1) < dropout_prob:
                condition_emb = None
        
        # Get loss based on scheduler type
        if self.use_edm:
            loss = self._compute_edm_loss(latents, condition_emb)
        else:
            # Sample timesteps for default diffusion
            t = torch.randint(0, self.diffusion.num_timesteps, (latents.shape[0],), device=self.device)

            # Get loss
            if condition_emb is not None:
                # Conditional training
                loss_dict = self.diffusion.training_losses(
                    lambda x, t: self.forward(x, t, condition_emb),
                    latents, t
                )
            else:
                # Unconditional training
                loss_dict = self.diffusion.training_losses(self.model, latents, t)

            loss = loss_dict["loss"].mean()
        
        # Enhanced logging
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # Avoid logging strings directly; log a numeric phase index instead
        try:
            phase_index = self.trainer.datamodule.phase_index if hasattr(self.trainer, 'datamodule') else 0
        except Exception:
            phase_index = 0
        self.log('phase_index', phase_index, on_step=False, on_epoch=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, on_epoch=False)
        
        # Log additional metrics
        if condition_emb is not None:
            self.log('condition_strength', condition_emb.norm().mean(), on_step=True, on_epoch=True)
        
        # Log batch size for monitoring
        self.log('batch_size', images.shape[0], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        
        # Similar to training but without optimization
        images = batch['image']
        
        with torch.no_grad():
            # Prepare inputs based on VAE configuration
            if self.vae is not None:
                # VAE latent space validation
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                
                latents = self.vae.encode(images).latent_dist.sample() * 0.18215
            else:
                # Pixel-level validation - use images directly
                latents = images
                # Ensure single channel for microscopy
                if latents.shape[1] == 3:
                    latents = latents[:, :1, :, :]  # Take first channel only
            
            condition_emb = self.encode_conditions(batch)
            
            # Use same loss computation as training
            if self.use_edm:
                loss = self._compute_edm_loss(latents, condition_emb)
            else:
                t = torch.randint(0, self.diffusion.num_timesteps, (latents.shape[0],), device=self.device)
                
                if condition_emb is not None:
                    loss_dict = self.diffusion.training_losses(
                        lambda x, t: self.forward(x, t, condition_emb),
                        latents, t
                    )
                else:
                    loss_dict = self.diffusion.training_losses(self.model, latents, t)
                
                loss = loss_dict["loss"].mean()
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log validation metrics
        self.log('val_batch_size', images.shape[0], on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        
        opt_config = self.config['optimizer']
        
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=opt_config.get('min_lr', 1e-6)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


class MicroscopyEvaluator:
    """Evaluation utilities for microscopy models"""
    
    def __init__(self, model: MicroscopyDiTModel):
        self.model = model
    
    @torch.no_grad()
    def compute_metrics(self, validation_loader):
        """Compute validation metrics"""
        
        total_loss = 0
        num_batches = 0
        
        self.model.eval()
        
        for batch in validation_loader:
            # Compute loss directly without Lightning logging
            images = batch['image']
            
            # Prepare inputs based on VAE configuration
            if self.model.vae is not None:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                latents = self.model.vae.encode(images).latent_dist.sample() * 0.18215
            else:
                latents = images
                if latents.shape[1] == 3:
                    latents = latents[:, :1, :, :]  # Take first channel only
            
            condition_emb = self.model.encode_conditions(batch)
            
            # Use same loss computation as training but without logging
            if self.model.use_edm:
                loss = self.model._compute_edm_loss(latents, condition_emb)
            else:
                t = torch.randint(0, self.model.diffusion.num_timesteps, (latents.shape[0],), device=self.model.device)
                if condition_emb is not None:
                    loss_dict = self.model.diffusion.training_losses(
                        lambda x, t: self.model.forward(x, t, condition_emb),
                        latents, t
                    )
                else:
                    loss_dict = self.model.diffusion.training_losses(self.model.model, latents, t)
                loss = loss_dict["loss"].mean()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'validation_loss': avg_loss,
            'num_samples': num_batches * validation_loader.batch_size
        }