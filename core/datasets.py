"""
Microscopy Dataset Loaders
==========================
Handles loading and pairing of foundation/degraded microscopy data
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F
from typing import Optional, Tuple
from typing import Union, Optional, Tuple
from typing import Dict, List, Union
import sys
import random
import numpy as np
from PIL import Image

# Import processed dataset from local utils
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from processed_dataset import build_manifest_dataset

# Local imports
from .conditioning import create_microscopy_conditions
from auto_degrade import auto_degrade_with_metadata


class MicroscopyDataset(Dataset):
    """Dataset for microscopy training with phase-aware loading"""
    
    def __init__(self, config: Dict, phase_config: Dict):
        self.config = config
        self.phase_config = phase_config
        self.condition_dropout = phase_config.get('condition_dropout', 0.0)
        # Optional resize transform based on config.model.input_size
        self._resize = None
        try:
            sz = tuple(self.config.get('model', {}).get('input_size') or [])
            if isinstance(sz, (list, tuple)) and len(sz) == 2:
                target_h, target_w = int(sz[0]), int(sz[1])
                def _transform(sample):
                    img = sample.get('image')
                    if torch.is_tensor(img) and img.dim() == 3:
                        if img.shape[-2:] != (target_h, target_w):
                            img = F.interpolate(img.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
                            sample['image'] = img
                    return sample
                self._resize = _transform
        except Exception:
            self._resize = None
        
        # Load datasets based on phase
        if phase_config['type'] == 'unconditional':
            self._setup_unconditional()
        else:
            self._setup_conditional()
    
    def _setup_unconditional(self):
        """Setup Phase 1: Foundation data only"""
        self.dataset = build_manifest_dataset(
            roots=[self.config['data']['foundation_path']],
            include_datasets=self.phase_config['datasets'],
            transform=self._resize
        )
        self.condition_types = []
        print(f"[DATASET] Phase 1: {len(self.dataset)} foundation samples")
    
    def _setup_conditional(self):
        """Setup Phase 2: Foundation + degraded pairs with weighted selection + manual degradation"""
        
        # Load foundation dataset
        self.foundation_dataset = build_manifest_dataset(
            roots=[self.config['data']['foundation_path']],
            include_datasets=self.phase_config['datasets'],
            transform=self._resize
        )
        
        # Check if manual degradation is enabled
        self.use_manual_degradation = self.phase_config.get('use_manual_degradation', False)
        self.manual_degradation_ratio = self.phase_config.get('manual_degradation_ratio', 0.6)
        
        if self.use_manual_degradation:
            # Load ALL foundation datasets for manual degradation
            self.manual_degradation_dataset = build_manifest_dataset(
                roots=[self.config['data']['foundation_path']],
                include_datasets=["sr_caco2", "fmd", "neuronal_cells", "rxrx1"]  # All foundation datasets
            )
            self.degradation_modes = self.phase_config.get('degradation_modes', ['blur', 'noise', 'downsample'])
            self.degradation_params = self.phase_config.get('degradation_params', {})
            self.degradation_to_task_map = self.phase_config.get('degradation_to_task_map', {
                'blur': 'deblurring',
                'noise': 'denoising', 
                'downsample': 'super_resolution'
            })
        
        # Load degraded datasets with proper naming (only 2 real degraded datasets exist)
        degraded_datasets = []
        for dataset_name in self.phase_config['datasets']:
            if dataset_name == 'sr_caco2':
                degraded_datasets.append('sr_caco2_lowres')
            elif dataset_name == 'fmd':
                degraded_datasets.append('fmd_noisy')
            # Only map existing degraded datasets
        
        self.degraded_dataset = build_manifest_dataset(
            roots=[self.config['data']['degraded_path']],
            include_datasets=degraded_datasets,
            transform=self._resize
        )
        
        # Setup weighted sampling for mixed real + manually degraded data
        self._setup_mixed_weighted_sampling()
        
        self.condition_types = self.phase_config.get('condition_types', [])
        
        manual_info = f" + {len(self.manual_degradation_dataset)} manual degradation samples" if self.use_manual_degradation else ""
        print(f"[DATASET] Phase 2: {len(self.foundation_dataset)} foundation + {len(self.degraded_dataset)} degraded{manual_info}")
        print(f"[CONDITIONS] Types: {self.condition_types}")
        if self.use_manual_degradation:
            print(f"[MANUAL DEGRADATION] Enabled with {self.manual_degradation_ratio:.1%} ratio")
            print(f"[DEGRADATION MODES] {self.degradation_modes}")
        print(f"[WEIGHTED SAMPLING] Mixed dataset weights configured")
    
    def _setup_mixed_weighted_sampling(self):
        """Setup weighted sampling for mixed real + manually degraded data (40% real, 60% manual)"""
        
        # Count samples per real degraded dataset
        real_dataset_counts = {}
        for sample in self.degraded_dataset.rows:
            dataset_name = sample.dataset
            real_dataset_counts[dataset_name] = real_dataset_counts.get(dataset_name, 0) + 1
        
        # Count samples per foundation dataset (for manual degradation)
        manual_dataset_counts = {}
        if self.use_manual_degradation:
            for sample in self.manual_degradation_dataset.rows:
                dataset_name = sample.dataset
                manual_dataset_counts[dataset_name] = manual_dataset_counts.get(dataset_name, 0) + 1
        
        # Calculate total effective samples
        total_real = sum(real_dataset_counts.values())
        total_manual = sum(manual_dataset_counts.values()) if self.use_manual_degradation else 0
        
        # Create combined sample weights
        self.sample_weights = []
        self.sample_types = []  # Track whether each sample is 'real' or 'manual'
        
        # Real degraded samples (40% of total weight)
        real_weight_per_dataset = 0.4 / len(real_dataset_counts) if real_dataset_counts else 0
        for sample in self.degraded_dataset.rows:
            dataset_name = sample.dataset
            dataset_size = real_dataset_counts[dataset_name]
            weight = real_weight_per_dataset / dataset_size if dataset_size > 0 else 0
            self.sample_weights.append(weight)
            self.sample_types.append('real')
        
        # Manual degraded samples (60% of total weight)
        if self.use_manual_degradation and manual_dataset_counts:
            manual_weight_per_dataset = 0.6 / len(manual_dataset_counts)
            for sample in self.manual_degradation_dataset.rows:
                dataset_name = sample.dataset
                dataset_size = manual_dataset_counts[dataset_name]
                weight = manual_weight_per_dataset / dataset_size if dataset_size > 0 else 0
                self.sample_weights.append(weight)
                self.sample_types.append('manual')
        
        print(f"[MIXED SAMPLING] Real datasets: {real_dataset_counts}")
        if self.use_manual_degradation:
            print(f"[MIXED SAMPLING] Manual datasets: {manual_dataset_counts}")
            print(f"[MIXED SAMPLING] Ratio: {1-self.manual_degradation_ratio:.1%} real + {self.manual_degradation_ratio:.1%} manual")
        
        # Normalize weights
        if self.sample_weights:
            weight_sum = sum(self.sample_weights)
            self.sample_weights = [w / weight_sum for w in self.sample_weights] if weight_sum > 0 else self.sample_weights
    
    def get_weighted_sampler(self):
        """Get WeightedRandomSampler for mixed real + manual degraded dataset"""
        if hasattr(self, 'sample_weights') and self.sample_weights:
            # Allow optional capping of samples per epoch
            max_samples = self.phase_config.get('max_samples_per_epoch', len(self.sample_weights))
            num_samples = min(len(self.sample_weights), int(max_samples))
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=num_samples,
                replacement=True
            )
        return None
    
    def __len__(self):
        if hasattr(self, 'foundation_dataset'):
            return len(self.foundation_dataset)
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.phase_config['type'] == 'unconditional':
            return self._get_unconditional_sample(idx)
        else:
            return self._get_conditional_sample(idx)
    
    def _get_unconditional_sample(self, idx):
        """Get Phase 1 sample: just images with descriptions"""
        sample = self.dataset[idx]
        
        # Create descriptive prompt
        description = self._create_description(sample)
        
        return {
            'image': sample['image'],
            'description': description
        }
    
    def _get_conditional_sample(self, idx):
        """Get Phase 2 sample: images + conditions (real degraded or manually degraded)"""
        
        # Get foundation sample
        foundation_sample = self.foundation_dataset[idx % len(self.foundation_dataset)]
        
        # Apply condition dropout (train unconditional capability)
        use_conditions = random.random() > self.condition_dropout
        
        # Create base return dict
        result = {
            'image': foundation_sample['image'],
            'description': self._create_description(foundation_sample)
        }
        
        if use_conditions:
            # Decide whether to use real degraded or manual degradation
            if self.use_manual_degradation and random.random() < self.manual_degradation_ratio:
                # Use manual degradation (60% of the time)
                self._add_manual_degraded_conditions(result, foundation_sample)
            else:
                # Use real degraded data (40% of the time, or 100% if manual degradation is disabled)
                self._add_real_degraded_conditions(result, foundation_sample)
        
        return result
    
    def _add_real_degraded_conditions(self, result, foundation_sample):
        """Add conditions from real degraded dataset"""
        condition_idx = 0
        
        # Map condition types to available real degraded datasets
        real_degraded_map = {
            'super_resolution': 'sr_caco2_lowres',  # low-res -> super_resolution task
            'denoising': 'fmd_noisy',               # noisy -> denoising task
            # Note: no real degraded dataset for deblurring task
        }
        
        for condition_type in self.condition_types:
            if condition_type in real_degraded_map:
                degraded_dataset_name = real_degraded_map[condition_type]
                degraded_sample = self._find_matching_degraded(foundation_sample, degraded_dataset_name)
                if degraded_sample:
                    result[f'condition_{condition_idx}'] = degraded_sample['image']
                    result[f'condition_type_{condition_idx}'] = condition_type
                    result[f'condition_strength_{condition_idx}'] = 1.0
                    condition_idx += 1
    
    def _add_manual_degraded_conditions(self, result, foundation_sample):
        """Add conditions from manually degraded foundation data"""
        condition_idx = 0
        
        # Convert foundation sample to numpy array if needed
        img_array = foundation_sample['image']
        if torch.is_tensor(img_array):
            img_array = img_array.cpu().numpy()
        
        # Create sample metadata dict from foundation sample
        sample_metadata = {
            'dataset': foundation_sample.get('dataset', ''),
            'NA': foundation_sample.get('NA', 1.0),
            'um_pixel': foundation_sample.get('um_pixel', 0.15),
            'lambda': foundation_sample.get('lambda', None),
            'marker': foundation_sample.get('marker', ''),
            'modality': foundation_sample.get('modality', 'WideField'),
            'channel_idx': foundation_sample.get('channel_idx', 0) if foundation_sample.get('dataset', '').startswith('rxrx1') else None
        }
        
        # Create reverse mapping from task to degradation mode using config
        task_to_degradation_map = {task: mode for mode, task in self.degradation_to_task_map.items()}
        # Provide a default mapping for tasks not explicitly configured (e.g., deblurring)
        defaults = {
            'deblurring': 'blur',
            'denoising': 'noise',
            'super_resolution': 'downsample',
        }
        for task, mode in defaults.items():
            task_to_degradation_map.setdefault(task, mode)
        
        for condition_type in self.condition_types:
            if condition_type in task_to_degradation_map:
                degradation_mode = task_to_degradation_map[condition_type]
                degraded_img, tag = self._apply_manual_degradation(img_array, sample_metadata, degradation_mode)
                result[f'condition_{condition_idx}'] = torch.from_numpy(degraded_img).float()
                result[f'condition_type_{condition_idx}'] = condition_type
                result[f'condition_strength_{condition_idx}'] = 1.0
                condition_idx += 1
    
    def _apply_manual_degradation(self, img_array, sample_metadata, degradation_mode):
        """Apply manual degradation with random parameters"""
        
        # Get parameter ranges from config
        params = self.degradation_params.get(degradation_mode, {})
        kwargs = {}
        
        if degradation_mode == 'blur':
            if 'psf_scale' in params:
                psf_range = params['psf_scale']
                kwargs['psf_scale'] = random.uniform(psf_range[0], psf_range[1])
            if 'extra_sigma_px' in params:
                extra_range = params['extra_sigma_px']
                kwargs['extra_sigma_px'] = random.uniform(extra_range[0], extra_range[1])
                
        elif degradation_mode == 'noise':
            if 'target_snr_at_full_scale' in params:
                snr_range = params['target_snr_at_full_scale']
                kwargs['target_snr_at_full_scale'] = random.uniform(snr_range[0], snr_range[1])
            if 'read_noise_e' in params:
                noise_range = params['read_noise_e']
                kwargs['read_noise_e'] = random.uniform(noise_range[0], noise_range[1])
                
        elif degradation_mode == 'downsample':
            if 'factor' in params:
                factor_range = params['factor']
                kwargs['factor'] = random.randint(int(factor_range[0]), int(factor_range[1]))
            kwargs['upsample_back'] = params.get('upsample_back', True)
        
        # Apply degradation
        return auto_degrade_with_metadata(img_array, sample_metadata, degradation_mode, **kwargs)
    
    def _find_matching_degraded(self, foundation_sample, degraded_dataset_name):
        """Find matching degraded sample for foundation sample"""
        
        # Simple approach: find sample from same base dataset
        foundation_dataset = foundation_sample['dataset']
        
        # Map foundation to degraded dataset
        target_dataset = None
        if foundation_dataset == 'sr_caco2' and degraded_dataset_name == 'sr_caco2_lowres':
            target_dataset = 'sr_caco2_lowres'
        elif foundation_dataset == 'fmd' and degraded_dataset_name == 'fmd_noisy':
            target_dataset = 'fmd_noisy'
        
        if not target_dataset:
            return None
        
        # Find samples from target dataset
        matching_samples = [
            sample for sample in self.degraded_dataset.rows
            if sample.dataset == target_dataset
        ]
        
        if matching_samples:
            # Return random matching sample
            selected = random.choice(matching_samples)
            return {
                'image': torch.load(selected.image) if selected.image.endswith('.pt') else 
                        torch.from_numpy(np.array(Image.open(selected.image))).float() / 255.0,
                'dataset': selected.dataset
            }
        
        return None
    
    def _create_description(self, sample):
        """Create descriptive prompt for sample"""
        
        components = ["high quality microscopy image"]
        
        # Add modality
        if sample.get('modality'):
            components.append(f"{sample['modality'].lower()} microscopy")
        
        # Add specimen info
        if sample.get('specimen'):
            specimen = sample['specimen'].replace('_', ' ').replace('-', ' ')
            components.append(f"of {specimen}")
        
        # Add marker info
        if sample.get('marker'):
            marker = sample['marker'].lower()
            components.append(f"showing {marker}")
        
        # Add task context
        if sample.get('task'):
            task_descriptions = {
                'super_resolution': 'high resolution detail',
                'denoising': 'clear structure',
                'segmentation': 'cellular boundaries',
                'classification': 'morphological features'
            }
            if sample['task'] in task_descriptions:
                components.append(f"with {task_descriptions[sample['task']]}")
        
        return ", ".join(components)


class ValidationDataset(Dataset):
    """Validation dataset for evaluation"""
    
    def __init__(self, foundation_path: str, datasets: List[str], max_samples: int = 100, target_size: Optional[Tuple[int, int]] = None):
        # Optional resize transform
        transform = None
        if target_size and len(target_size) == 2:
            th, tw = int(target_size[0]), int(target_size[1])
            def _t(sample):
                img = sample.get('image')
                if torch.is_tensor(img) and img.dim() == 3 and img.shape[-2:] != (th, tw):
                    img = F.interpolate(img.unsqueeze(0), size=(th, tw), mode='bilinear', align_corners=False).squeeze(0)
                    sample['image'] = img
                return sample
            transform = _t
        self.dataset = build_manifest_dataset(
            roots=[foundation_path],
            include_datasets=datasets,
            transform=transform
        )
        
        # Limit validation size
        if len(self.dataset) > max_samples:
            indices = torch.randperm(len(self.dataset))[:max_samples]
            self.dataset.rows = [self.dataset.rows[i] for i in indices]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'image': sample['image'],
            'description': f"microscopy {sample['dataset']} validation"
        }