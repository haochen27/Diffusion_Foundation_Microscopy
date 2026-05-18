"""
Image Quality Metrics
=====================
PSNR and SSIM computation for microscopy reconstruction evaluation.
All inputs expected in [-1, 1] range. Internally converted to [0, 1].
"""

import torch
import torch.nn.functional as F


def to_01(x):
    """Convert [-1, 1] to [0, 1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)


def compute_psnr(recon, gt, data_range=1.0):
    """
    Compute PSNR between reconstructed and ground truth images.
    Args:
        recon: Reconstructed image in [-1, 1], shape [1, 1, H, W] or [1, H, W]
        gt: Ground truth image in [-1, 1], same shape
        data_range: Maximum pixel value after conversion (1.0 for [0, 1])
    Returns:
        PSNR value in dB (float)
    """
    recon_01 = to_01(recon).float()
    gt_01 = to_01(gt).float()
    mse = F.mse_loss(recon_01, gt_01)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(data_range ** 2 / mse)).item()


def compute_ssim(recon, gt, window_size=11, data_range=1.0):
    """
    Compute SSIM between reconstructed and ground truth images.
    Implements Wang et al. 2004 (simplified, single-scale).
    Args:
        recon: Reconstructed image in [-1, 1], shape [B, C, H, W] or [C, H, W]
        gt: Ground truth image in [-1, 1], same shape
        window_size: Size of Gaussian window
        data_range: Maximum pixel value (1.0 for [0, 1])
    Returns:
        SSIM value (float in [0, 1])
    """
    recon_01 = to_01(recon).float()
    gt_01 = to_01(gt).float()

    # Ensure 4D
    if recon_01.dim() == 3:
        recon_01 = recon_01.unsqueeze(0)
        gt_01 = gt_01.unsqueeze(0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    channel = recon_01.shape[1]
    coords = torch.arange(window_size, dtype=torch.float32, device=recon_01.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)  # [K, K]
    window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)  # [C, 1, K, K]

    pad = window_size // 2

    mu1 = F.conv2d(recon_01, window, padding=pad, groups=channel)
    mu2 = F.conv2d(gt_01, window, padding=pad, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(recon_01 ** 2, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt_01 ** 2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(recon_01 * gt_01, window, padding=pad, groups=channel) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_metrics(recon, gt):
    """
    Compute all metrics between reconstructed and ground truth.
    Args:
        recon: [-1, 1] tensor
        gt: [-1, 1] tensor
    Returns:
        dict with 'psnr' and 'ssim' keys
    """
    return {
        'psnr': compute_psnr(recon, gt),
        'ssim': compute_ssim(recon, gt)
    }
