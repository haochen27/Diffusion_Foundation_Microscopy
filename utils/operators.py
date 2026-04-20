"""
Forward Operators for Posterior Sampling
==========================================
Measurement operators (blur, noise, etc.) for inverse problems
"""

import torch
import numpy as np


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

    def measure_poisson(self, x, scale=100.0):
        """Measure input by applying operator and adding Poisson noise"""
        self.poisson_scale = scale
        y0 = self(x)
        rate = torch.clamp((y0+1) / 2 * 1, min=0)
        noisy_counts = torch.poisson(rate*scale)
        return (noisy_counts/scale*2-1).clamp(-1, 1)

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
