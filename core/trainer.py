"""
Microscopy Training Pipeline
===========================
Main training orchestrator with phase management using DiT-XL/8
"""

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
import os

# Local imports
from .models import MicroscopyDiTModel, MicroscopyEvaluator
from .unet.model import MicroscopyUnetModel
from .datasets import MicroscopyDataset, ValidationDataset
from .phase_manager import MicroscopyPhaseManager
from .callbacks import (
    EMACallback,
    RealTimeMonitorCallback,
    SampleGenerationCallback,
    CustomCheckpointCallback,
    ImageLoggingCallback,
)


class MicroscopyTrainer:
    """Main training orchestrator for DiT-based microscopy diffusion"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.phase_manager = MicroscopyPhaseManager(config)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Initialize logging without wandb dependency"""
        pass
    
    def train_phase(self, phase_config: Dict):
        """Train a single phase"""
        
        print(f"\n[PHASE] Starting: {phase_config['name']}")
        print(f"[INFO] Type: {phase_config['type']}")
        print(f"[INFO] Epochs: {phase_config['epochs']}")
        
        # Create model based on architecture
        architecture = str(self.config.get('model', {}).get('architecture', 'DiT-S/8'))
        if architecture.lower() == 'unet':
            model = MicroscopyUnetModel(
                config=self.config,
                phase_config=phase_config
            )
        else:
            model = MicroscopyDiTModel(
                config=self.config,
                phase_config=phase_config
            )
        
        # Setup data
        train_dataset = MicroscopyDataset(self.config, phase_config)
        
        # Get weighted sampler for conditional phases to balance degraded datasets
        sampler = None
        shuffle = True
        if phase_config['type'] == 'conditional':
            sampler = train_dataset.get_weighted_sampler()
            shuffle = False if sampler else True  # Don't shuffle when using sampler
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=phase_config['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config['train']['dataloader_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        # Validation data
        # Use model.input_size for validation resize if provided
        target_sz = tuple(self.config.get('model', {}).get('input_size') or [])
        if not (isinstance(target_sz, (list, tuple)) and len(target_sz) == 2):
            target_sz = None
        val_dataset = ValidationDataset(
            self.config['data']['foundation_path'],
            phase_config['datasets'],
            max_samples=50,
            target_size=target_sz
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2
        )
        
        # Setup callbacks
        callbacks = self.setup_callbacks(phase_config)
        
        # Setup loggers
        loggers = self.setup_phase_loggers(phase_config)
        
        # Create trainer with enhanced monitoring
        # Choose accelerator/precision optimized for A100/H100
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        # Prefer bf16 on Ampere/Hopper, else fallback to amp fp16; CPU uses 32
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                precision = 'bf16-mixed'
            else:
                precision = '16-mixed'
        else:
            precision = '32-true'

        max_steps = phase_config.get('max_steps')
        min_steps = phase_config.get('min_steps')
        trainer = L.Trainer(
            max_epochs=phase_config['epochs'],
            max_steps=max_steps,
            min_steps=min_steps,
            callbacks=callbacks,
            logger=loggers,
            accelerator=accelerator,
            devices='auto',
            # Use DDP with static graph if 2-4 GPUs; else auto
            strategy=('ddp' if 1 < torch.cuda.device_count() <= 4 else (
                'ddp_find_unused_parameters_true' if torch.cuda.device_count() > 4 else 'auto'
            )),
            precision=precision,
            accumulate_grad_batches=self.config['train']['accumulate_grad_batches'],
            log_every_n_steps=self.config.get('monitoring', {}).get('log_every_n_steps', 10),
            val_check_interval=self.config.get('monitoring', {}).get('val_check_interval', 100),
            enable_progress_bar=self.config.get('monitoring', {}).get('progress_bar', True),
            gradient_clip_val=1.0,
            enable_checkpointing=True,
            detect_anomaly=False,  # Disable for performance
            benchmark=True  # Enable cudnn benchmark
        )
        
        # Determine checkpoint path for resume
        resume_checkpoint = None
        
        # First check config for manual checkpoint (works regardless of --resume flag)
        if phase_config.get('resume_from_checkpoint'):
            config_checkpoint = phase_config['resume_from_checkpoint']
            if os.path.exists(config_checkpoint):
                resume_checkpoint = config_checkpoint
                print(f"[RESUME] Using config checkpoint: {resume_checkpoint}")
            else:
                print(f"[WARNING] Config checkpoint not found: {config_checkpoint}")
        elif hasattr(self, '_resume') and self._resume:
            # Auto-discover latest checkpoint for this phase only if --resume flag used
            discovered_checkpoint = self.phase_manager.find_latest_checkpoint(phase_config['name'])
            if discovered_checkpoint and os.path.exists(discovered_checkpoint):
                resume_checkpoint = discovered_checkpoint
                print(f"[RESUME] Auto-discovered checkpoint: {resume_checkpoint}")
            elif discovered_checkpoint:
                print(f"[WARNING] Auto-discovered checkpoint not found: {discovered_checkpoint}")
        
        # Train (handle weights-only checkpoints gracefully)
        if resume_checkpoint is not None:
            try:
                ckpt_meta = torch.load(resume_checkpoint, map_location='cpu')
                # Treat as full state ONLY if optimizer_states exist and non-empty
                is_full_state = (
                    isinstance(ckpt_meta, dict)
                    and ('optimizer_states' in ckpt_meta)
                    and isinstance(ckpt_meta.get('optimizer_states'), (list, tuple))
                    and len(ckpt_meta.get('optimizer_states')) > 0
                )
            except Exception:
                is_full_state = False

            if not is_full_state:
                # We have a weights-only checkpoint: load weights into the model and start fresh
                try:
                    state_dict = ckpt_meta['state_dict'] if isinstance(ckpt_meta, dict) and 'state_dict' in ckpt_meta else ckpt_meta
                    model.load_state_dict(state_dict, strict=False)
                    print(f"[RESUME] Loaded weights-only checkpoint into model (no optimizer state): {resume_checkpoint}")
                except Exception as e:
                    print(f"[WARNING] Failed loading weights-only checkpoint '{resume_checkpoint}': {e}")
                resume_ckpt_for_fit = None
            else:
                resume_ckpt_for_fit = resume_checkpoint
        else:
            resume_ckpt_for_fit = None

        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt_for_fit)
        
        # Get latest checkpoint saved by CustomCheckpointCallback or ModelCheckpoint
        best_checkpoint = None
        # Prefer best, fall back to last
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                if getattr(cb, "best_model_path", None):
                    best_checkpoint = cb.best_model_path
                elif getattr(cb, "last_model_path", None):
                    best_checkpoint = cb.last_model_path
            if isinstance(cb, CustomCheckpointCallback):
                # Custom callback saves at fixed epochs; take the most recent file if exists
                # We can't easily read its internals, so default to trainer.checkpoint_callback if available
                pass
        
        # If EMA is used, validate with EMA weights
        try:
            ema_cb = next(cb for cb in callbacks if getattr(cb, "name", "") == "EMA" or cb.__class__.__name__ == "EMACallback")
        except StopIteration:
            ema_cb = None
        if ema_cb is not None and hasattr(ema_cb, "apply_ema_weights") and hasattr(ema_cb, "restore_original_weights"):
            ema_cb.apply_ema_weights(model)
        evaluator = MicroscopyEvaluator(model)
        final_metrics = evaluator.compute_metrics(val_loader)
        if ema_cb is not None and hasattr(ema_cb, "restore_original_weights"):
            ema_cb.restore_original_weights(model)
        
        # Complete phase
        self.phase_manager.complete_phase(best_checkpoint, final_metrics)
        
        return best_checkpoint
    
    def setup_callbacks(self, phase_config: Dict):
        """Setup enhanced training callbacks with EMA and monitoring"""
        
        callbacks = []
        
        # Sample generation callback with EDM support (place BEFORE checkpoints so sampling runs first)
        if self.config.get('monitoring', {}).get('plot_samples', True):
            monitoring_config = self.config.get('monitoring', {})
            sample_callback = SampleGenerationCallback(
                sample_every_n_epochs=monitoring_config.get('sample_every_n_epochs', 5),
                num_samples=monitoring_config.get('num_samples', 4),
                save_dir=str(Path(self.config['train']['save_path']) / 'samples' / phase_config['name']),
                edm_steps=monitoring_config.get('edm_steps', 100),
                start_epoch=monitoring_config.get('sample_start_epoch', 10)
            )
            callbacks.append(sample_callback)
        
        # Custom checkpoint callback with epochs_to_save
        checkpoint_dir = Path(self.config['train']['save_path']) / 'checkpoints'
        epochs_to_save = phase_config.get('epochs_to_save', [phase_config['epochs']])
        
        custom_checkpoint = CustomCheckpointCallback(
            save_dir=str(checkpoint_dir),
            epochs_to_save=epochs_to_save,
            phase_name=phase_config['name']
        )
        callbacks.append(custom_checkpoint)
        
        # Also track best val_loss via ModelCheckpoint if validation exists
        try:
            monitor_metric = self.config.get('monitoring', {}).get('monitor_metric', 'val_loss')
            model_ckpt = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename=f"{phase_config['name']}_best-{{epoch:03d}}",
                monitor=monitor_metric,
                mode='min',
                save_top_k=1,
                save_last=True,
                save_weights_only=True
            )
            callbacks.append(model_ckpt)
        except Exception as e:
            print(f"[WARNING] ModelCheckpoint setup failed: {e}")

        # EMA callback
        if self.config.get('ema', {}).get('enabled', True):
            ema_callback = EMACallback(
                decay=self.config['ema'].get('decay', 0.9999),
                update_every=self.config['ema'].get('update_every', 1),
                start_step=self.config['ema'].get('start_step', 2000)
            )
            callbacks.append(ema_callback)
        
        # Real-time monitoring
        if self.config.get('monitoring', {}).get('log_every_n_steps'):
            monitor_callback = RealTimeMonitorCallback(
                log_every_n_steps=self.config['monitoring']['log_every_n_steps'],
                plot_loss=self.config['monitoring'].get('plot_loss', True),
                plot_lr=self.config['monitoring'].get('plot_lr', True),
                save_plots=True,
                plot_dir=str(Path(self.config['train']['save_path']) / 'plots' / phase_config['name']),
                track_throughput=True,
                track_gpu_memory=True,
                track_grad_norm=True,
            )
            callbacks.append(monitor_callback)
        
        # Image logging callback for inputs/conditions
        img_log_steps = int(self.config.get('monitoring', {}).get('image_log_every_n_steps', 0) or 0)
        if img_log_steps > 0:
            callbacks.append(
                ImageLoggingCallback(
                    log_every_n_steps=img_log_steps,
                    max_images=int(self.config.get('monitoring', {}).get('image_log_max_images', 8) or 8),
                    save_dir=str(Path(self.config['train']['save_path']) / 'images' / phase_config['name'])
                )
            )
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(
            logging_interval='step',
            log_momentum=False
        )
        callbacks.append(lr_monitor)
        
        
        return callbacks
    
    def setup_phase_loggers(self, phase_config: Dict):
        """Setup phase-specific loggers"""
        
        loggers = []
        
        # WandB logger (optional)
        if self.config['train'].get('wandb'):
            try:
                wandb_logger = WandbLogger(
                    project=self.config['train']['wandb']['project'],
                    name=f"{phase_config['name']}_{self.config.get('run_id', 'default')}",
                    save_dir=str(Path(self.config['train']['save_path']) / 'logs')
                )
                loggers.append(wandb_logger)
            except Exception as e:
                print(f"[WARNING] WandB setup failed: {e}")
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(Path(self.config['train']['save_path']) / 'logs'),
            name=phase_config['name']
        )
        loggers.append(tb_logger)
        
        return loggers
    
    def run_training_pipeline(self, resume=False):
        """Run complete training pipeline"""
        
        print("[PIPELINE] Microscopy Diffusion Training Pipeline")
        print(f"[INFO] Save path: {self.config['train']['save_path']}")
        print(f"[INFO] Total phases: {len(self.config['phases'])}")
        
        # Store resume flag
        self._resume = resume
        
        # Log pipeline start
        self.phase_manager.log_event("Training pipeline started")
        
        while not self.phase_manager.is_complete():
            phase_config = self.phase_manager.get_current_phase()
            
            if not phase_config:
                break
            
            # Skip if already completed
            if phase_config['name'] in self.phase_manager.state['completed_phases']:
                print(f"[SKIP] Already completed: {phase_config['name']}")
                self.phase_manager.state['current_phase'] += 1
                continue
            
            # Train phase
            try:
                best_checkpoint = self.train_phase(phase_config)
                
                # Log transition
                self.phase_manager.log_event(
                    f"Phase {phase_config['name']} completed",
                    {"checkpoint": best_checkpoint}
                )
                
                # Check for next phase
                if not self.phase_manager.is_complete():
                    next_phase = self.phase_manager.get_current_phase()
                    if next_phase:
                        print(f"[TRANSITION] Auto-transitioning to: {next_phase['name']}")
                        self.phase_manager.log_event(f"Transitioning to: {next_phase['name']}")
                
            except Exception as e:
                error_msg = f"Error in {phase_config['name']}: {e}"
                print(f"[ERROR] {error_msg}")
                self.phase_manager.log_event(error_msg, {"error": str(e)})
                raise
        
        # Pipeline complete
        print("[COMPLETE] All phases completed successfully!")
        summary = self.phase_manager.get_training_summary()
        print(f"[SUMMARY] {summary}")
        
        self.phase_manager.log_event("Training pipeline completed", summary)
        
        return summary