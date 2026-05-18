"""
Baseline Deconvolution (No Diffusion Prior)
============================================
Pure algorithmic baselines:
  - MCMC (Langevin): For Gaussian noise (posterior_samples)
  - Richardson-Lucy + TV: For Poisson noise (posterior_samples_RL)

Uses the same images, blur kernel, noise levels, and saving format
as posterior_sample.py and posterior_sample_RL.py.
"""

import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
import argparse
import random
from torchvision.utils import save_image
from utils.metrics import compute_metrics


# ======================== Image I/O ========================

def load_image(image_path):
    """Load a single-channel image and normalize to [-1, 1]"""
    img = Image.open(image_path)
    if img.mode == 'I;16':
        img_arr = np.array(img, dtype=np.float32) / 65535.0
    else:
        img = img.convert('L')
        img_arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)
    return tensor * 2.0 - 1.0


def save_results(images_dict, output_path):
    """Save side-by-side comparison of all provided images (all in [-1, 1])."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for name, img in images_dict.items():
        if img is not None:
            panels.append((img * 0.5 + 0.5).clamp(0, 1))
    if len(panels) == 0:
        return
    side_by_side = torch.cat(panels, dim=2)
    save_image(side_by_side, str(output_path))
    print(f"  [SAVE] {output_path}")


# ======================== Operators ========================

def create_blur_operator(kernel_size=31, intensity=5.0, device='cuda'):
    """Create Gaussian blur as nn.Sequential (matches GaussianBlur operator)."""
    import scipy.ndimage
    import torch.nn as nn

    n = np.zeros((kernel_size, kernel_size))
    n[kernel_size // 2, kernel_size // 2] = 1
    k = scipy.ndimage.gaussian_filter(n, sigma=intensity)
    k_tensor = torch.from_numpy(k).float()

    conv = nn.Sequential(
        nn.ReflectionPad2d(kernel_size // 2),
        nn.Conv2d(1, 1, kernel_size, stride=1, padding=0, bias=False, groups=1)
    )
    conv[1].weight.data = k_tensor.unsqueeze(0).unsqueeze(0)
    for param in conv.parameters():
        param.requires_grad = False
    conv = conv.to(device)
    return conv


def apply_blur(conv, x):
    """Apply blur operator."""
    return conv(x)


def measure_gaussian(conv, x, noise_sigma=0.1):
    """Blur + additive Gaussian noise."""
    y0 = apply_blur(conv, x)
    return y0 + noise_sigma * torch.randn_like(y0)


def measure_poisson(conv, x, scale=100.0):
    """Blur + Poisson noise (matches operators.py)."""
    y0 = apply_blur(conv, x)
    rate = torch.clamp((y0 + 1) / 2, min=0)
    noisy_counts = torch.poisson(rate * scale)
    return (noisy_counts / scale * 2 - 1).clamp(-1, 1)


# ======================== TV Regularization ========================

def compute_tv_gradient(x):
    """Compute gradient of isotropic Total Variation."""
    eps = 1e-8
    diff_x = torch.zeros_like(x)
    diff_y = torch.zeros_like(x)
    diff_x[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
    diff_y[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]

    # Isotropic TV norm
    norm_x = torch.sqrt(diff_x[:, :, :, :-1]**2 + diff_y[:, :, :, :-1]**2 + eps)
    norm_y = torch.sqrt(diff_x[:, :, :-1, :]**2 + diff_y[:, :, :-1, :]**2 + eps)

    tv_grad = torch.zeros_like(x)
    tv_grad[:, :, :, 1:]  += diff_x[:, :, :, :-1] / norm_x
    tv_grad[:, :, 1:, :]  += diff_y[:, :, :-1, :] / norm_y
    tv_grad[:, :, :, :-1] -= diff_x[:, :, :, :-1] / norm_x
    tv_grad[:, :, :-1, :] -= diff_y[:, :, :-1, :] / norm_y
    return tv_grad


# ======================== MCMC Deconvolution (Gaussian) ========================

def mcmc_deconvolution(conv, measurement, num_iters=500, lr=0.01, tv_weight=0.001):
    """
    Pure Langevin MCMC deconvolution for Gaussian noise.
    Minimizes: ||K*x - y||^2 + tv_weight * TV(x)
    No diffusion prior.
    """
    x = measurement.clone().detach()  # Initialize with measurement

    for i in range(num_iters):
        x_var = x.clone().detach().requires_grad_(True)

        # Data fidelity: ||K*x - y||^2
        residual = apply_blur(conv, x_var) - measurement
        data_loss = (residual ** 2).sum()

        data_grad = torch.autograd.grad(data_loss, x_var)[0]

        with torch.no_grad():
            # TV gradient
            tv_grad = compute_tv_gradient(x)

            # Langevin update
            noise = torch.randn_like(x) * np.sqrt(2 * lr)
            x = x - lr * (data_grad + tv_weight * tv_grad) + noise
            x = x.clamp(-1, 1)

        if (i + 1) % 100 == 0:
            print(f"    MCMC iter {i+1}/{num_iters} | data_loss={data_loss.item():.4f}")

    return x.detach()


# ======================== RL Deconvolution (Poisson) ========================

def rl_deconvolution(conv, measurement, scale, num_iters=200, tv_weight=0.001, alpha=0.5):
    """
    Richardson-Lucy deconvolution with TV regularization for Poisson noise.
    No diffusion prior.
    """
    eps = 1e-8

    # Map to positive domain [0, inf)
    y = torch.clamp((measurement + 1.0) * 0.5, min=0.0) * 255.0
    x = torch.clamp((measurement + 1.0) * 0.5, min=eps) * 255.0  # Init with measurement

    for i in range(num_iters):
        with torch.no_grad():
            # Forward model K*x
            x_norm = (x / 255.0).clamp(0, 1) * 2.0 - 1.0
            Kx = apply_blur(conv, x_norm)
            Kx = torch.clamp((Kx + 1.0) * 0.5, min=eps) * 255.0

            # RL ratio
            ratio = y / Kx

            # Back-project ratio through K^T (blur is symmetric so K^T = K)
            ratio_norm = ratio * 2.0 - 1.0
            smoothed = apply_blur(conv, ratio_norm)
            mult_factor = (smoothed + 1.0) * 0.5
            mult_factor = mult_factor.clamp(0.1, 10.0)

            # TV regularization
            tv_grad = compute_tv_gradient(x_norm)

            # Multiplicative RL update with TV (Dey-Sutour form: divide by 1 + λ ∇TV)
            x = x * (mult_factor ** alpha) / (1.0 + tv_weight * tv_grad).clamp(min=0.1)

        if (i + 1) % 50 == 0:
            print(f"    RL iter {i+1}/{num_iters}")

    # Map back to [-1, 1]
    result = (x / 255.0).clamp(0, 1) * 2.0 - 1.0
    return result


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(description="Baseline deconvolution (no diffusion prior)")
    parser.add_argument('--mode', type=str, required=True, choices=['mcmc', 'rl'],
                        help='mcmc = Gaussian noise, rl = Poisson noise')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file (uses posterior_sampling.yaml or posterior_sampling_RL.yaml by default)')
    parser.add_argument('--num_iters', type=int, default=500,
                        help='Number of optimization iterations')
    parser.add_argument('--tv_weight', type=float, default=0.001,
                        help='TV regularization weight')
    parser.add_argument('--poisson_scale', type=int, default=None,
                        help='Override Poisson scale (RL mode only)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Override number of samples')
    parser.add_argument('--save_comparison', action='store_true', default=True,
                        help='Save measurement|recon|GT comparison panel (default: True)')
    parser.add_argument('--no_save_comparison', dest='save_comparison', action='store_false',
                        help='Save only reconstruction image')
    args = parser.parse_args()

    # Load config
    if args.config is None:
        if args.mode == 'mcmc':
            args.config = './config/posterior_sampling.yaml'
        else:
            args.config = './config/posterior_sampling_RL.yaml'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = config.get('device', {}).get('device', 'cuda')
    seed = config.get('advanced', {}).get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Operator
    op_config = config.get('operator', {})
    kernel_size = op_config.get('kernel_size', 31)
    intensity = op_config.get('intensity', 5.0)
    conv = create_blur_operator(kernel_size, intensity, device)

    # Data
    data_config = config.get('data', {})
    test_dir = Path(data_config.get('test_dir', './microscopy_runs_unet/test_data'))
    num_samples = args.num_samples or data_config.get('num_samples', 10)
    random_select = data_config.get('random_select', True)

    all_images = sorted(list(test_dir.glob('*.png')))
    if random_select:
        random.shuffle(all_images)
    image_files = all_images[:num_samples]

    print(f"Mode: {args.mode.upper()}")
    print(f"Images: {len(image_files)} from {test_dir}")
    print(f"Blur: kernel={kernel_size}, intensity={intensity}")
    print(f"Iterations: {args.num_iters}, TV weight: {args.tv_weight}")

    if args.mode == 'mcmc':
        # ========== Gaussian MCMC baseline ==========
        noise_sigma = op_config.get('noise_sigma', 0.1)
        base_output_dir = Path(config.get('output', {}).get('save_dir', './posterior_samples'))
        output_dir = base_output_dir / 'MCMC'
        output_dir.mkdir(parents=True, exist_ok=True)

        lr = config.get('daps', {}).get('mcmc_lr', 0.01)
        print(f"Noise sigma: {noise_sigma}, MCMC lr: {lr}")
        print(f"Output: {output_dir}\n")

        all_metrics = []
        for idx, img_path in enumerate(image_files):
            print(f"[{idx+1}/{len(image_files)}] {img_path.name}")
            clean = load_image(img_path).to(device)

            with torch.no_grad():
                measurement = measure_gaussian(conv, clean, noise_sigma)
                measurement = measurement.clamp(-1, 1)

            recon = mcmc_deconvolution(conv, measurement,
                                       num_iters=args.num_iters,
                                       lr=lr,
                                       tv_weight=args.tv_weight)

            save_results({
                'measurement': measurement[0].cpu(),
                'reconstructed': recon[0].cpu(),
                'GT': clean[0].cpu()
            }, output_dir / f'result_{idx:04d}_{img_path.stem}.png')

            metrics = compute_metrics(recon[0].cpu(), clean[0].cpu())
            input_m = compute_metrics(measurement[0].cpu(), clean[0].cpu())
            all_metrics.append({'image': img_path.name,
                                'recon_psnr': metrics['psnr'], 'recon_ssim': metrics['ssim'],
                                'input_psnr': input_m['psnr'], 'input_ssim': input_m['ssim']})
            print(f"  Input   PSNR: {input_m['psnr']:.2f} dB | SSIM: {input_m['ssim']:.4f}")
            print(f"  Recon   PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")

        # Save metrics summary
        if all_metrics:
            avg = {k: np.mean([m[k] for m in all_metrics]) for k in ['recon_psnr', 'recon_ssim', 'input_psnr', 'input_ssim']}
            summary = {'method': 'Baseline_MCMC', 'per_image': all_metrics, 'average': avg}
            with open(output_dir / 'metrics_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n{'='*60}")
            print(f"  Input   Avg PSNR: {avg['input_psnr']:.2f} dB | SSIM: {avg['input_ssim']:.4f}")
            print(f"  Recon   Avg PSNR: {avg['recon_psnr']:.2f} dB | SSIM: {avg['recon_ssim']:.4f}")
            print(f"{'='*60}")

    else:
        # ========== RL baseline ==========
        poisson_config = config.get('poisson', {})
        scales = [args.poisson_scale] if args.poisson_scale else poisson_config.get('scales', [10])
        rl_alpha = poisson_config.get('rl_alpha', 0.5)
        base_output_dir = Path(config.get('output', {}).get('save_dir', './posterior_samples_RL'))

        for scale in scales:
            output_dir = base_output_dir / f'scale_{scale}' / 'RL'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"Poisson scale: {scale}")
            print(f"Output: {output_dir}")
            print(f"{'='*60}")

            all_metrics = []
            for idx, img_path in enumerate(image_files):
                print(f"[{idx+1}/{len(image_files)}] {img_path.name}")
                clean = load_image(img_path).to(device)

                with torch.no_grad():
                    measurement = measure_poisson(conv, clean, scale)

                recon = rl_deconvolution(conv, measurement, scale,
                                         num_iters=args.num_iters,
                                         tv_weight=args.tv_weight,
                                         alpha=rl_alpha)

                out_path = output_dir / f'result_{idx:04d}_{img_path.stem}.png'
                if args.save_comparison:
                    save_results({
                        'measurement': measurement[0].cpu(),
                        'reconstructed': recon[0].cpu(),
                        'GT': clean[0].cpu()
                    }, out_path)
                else:
                    recon_01 = (recon[0].cpu() * 0.5 + 0.5).clamp(0, 1)
                    save_image(recon_01, str(out_path))
                    print(f"  [SAVE] {out_path}")

                metrics = compute_metrics(recon[0].cpu(), clean[0].cpu())
                input_m = compute_metrics(measurement[0].cpu(), clean[0].cpu())
                all_metrics.append({'image': img_path.name,
                                    'recon_psnr': metrics['psnr'], 'recon_ssim': metrics['ssim'],
                                    'input_psnr': input_m['psnr'], 'input_ssim': input_m['ssim']})
                print(f"  Input   PSNR: {input_m['psnr']:.2f} dB | SSIM: {input_m['ssim']:.4f}")
                print(f"  Recon   PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")

            # Save metrics summary
            if all_metrics:
                avg = {k: np.mean([m[k] for m in all_metrics]) for k in ['recon_psnr', 'recon_ssim', 'input_psnr', 'input_ssim']}
                summary = {'method': f'Baseline_RL_scale_{int(scale)}', 'per_image': all_metrics, 'average': avg}
                with open(output_dir / 'metrics_summary.json', 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"\n{'='*60}")
                print(f"  Input   Avg PSNR: {avg['input_psnr']:.2f} dB | SSIM: {avg['input_ssim']:.4f}")
                print(f"  Recon   Avg PSNR: {avg['recon_psnr']:.2f} dB | SSIM: {avg['recon_ssim']:.4f}")
                print(f"{'='*60}")

    print(f"\n[DONE] All baseline results saved.")


if __name__ == '__main__':
    main()
