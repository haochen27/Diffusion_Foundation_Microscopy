#!/usr/bin/env python3
"""
Microscopy Diffusion Training - DiT-XL/8 Based Pipeline
=======================================================
Automated phase training for 512x512 microscopy generation

Usage:
    python train.py --config config/microscopy.yaml
"""

import torch
import argparse
import yaml
from pathlib import Path
import sys
import csv
import json
import numpy as np
from torch.utils import tensorboard

# Add repo and core module paths for absolute imports
repo_dir = Path(__file__).parent
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(repo_dir / 'core'))
from core.trainer import MicroscopyTrainer
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from core.callbacks import RealTimeMonitorCallback, ImageLoggingCallback
from core.models import MicroscopyDiTModel


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add run ID for tracking
    config['run_id'] = f"run_{torch.randint(1000, 9999, (1,)).item()}"
    
    return config


def main():
    """Main training entry point"""
    
    parser = argparse.ArgumentParser(description="Microscopy Diffusion Training")
    parser.add_argument('--config', type=str, required=True, help='Configuration YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    # Smoke test removed for production training
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Performance knobs for A100/H100
    try:
        torch.set_float32_matmul_precision('high')
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    print("[INIT] Microscopy Diffusion Training Pipeline")
    print(f"[CONFIG] {args.config}")
    print(f"[RUN_ID] {config['run_id']}")
    
    # No smoke test path in production mode
    
    # Create trainer and run pipeline
    trainer = MicroscopyTrainer(config)
    summary = trainer.run_training_pipeline(resume=args.resume)
    print(f"[FINAL] Training completed: {summary}")


if __name__ == "__main__":
    main()