"""
MCMC Sampler for Posterior Sampling
====================================
Langevin dynamics and HMC for data consistency in DAPS
"""

import torch
import numpy as np
import tqdm


class MCMCSampler:
    """MCMC sampler for posterior sampling"""

    def __init__(self, num_steps=10, lr=0.1, tau=0.01, lr_min_ratio=0.01,
                 prior_solver='gaussian', mc_algo='langevin', momentum=0.9):
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.mc_algo = mc_algo
        self.momentum = momentum

    def score_fn(self, x, x0hat, model, xt, operator, measurement, sigma):
        """Compute conditional score function (data-fitting term only)."""
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
        data_term = -data_fitting_grad / self.tau ** 2

        total_score = torch.clamp(data_term, -100.0, 100.0)

        return total_score, data_fitting_loss

    def get_prior_score(self, x, x0hat, xt, model, sigma):
        """Get prior score based on solver type"""
        if self.prior_solver == 'gaussian':
            return (x0hat - xt).detach() / sigma ** 2
        else:
            raise NotImplementedError(f"Prior solver {self.prior_solver} not implemented")

    def prepare_prior_score(self, x0hat, xt, model, sigma):
        """Precompute prior score"""
        if self.prior_solver == 'gaussian':
            self.prior_score = (x0hat - xt).detach() / sigma ** 2

    def mc_prepare(self, x0hat, xt, model, operator, measurement, sigma):
        """Prepare MCMC state"""
        if self.mc_algo == 'hmc':
            self.velocity = torch.randn_like(x0hat)

    def mc_update(self, x, cur_score, lr, epsilon):
        """Perform MCMC update step"""
        if self.mc_algo == 'langevin':
            x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
        elif self.mc_algo == 'hmc':
            step_size = np.sqrt(lr)
            self.velocity = self.momentum * self.velocity + step_size * cur_score + \
                          np.sqrt(2 * (1 - self.momentum)) * epsilon
            x_new = x + self.velocity * step_size
        else:
            raise NotImplementedError
        return x_new

    def get_lr(self, ratio):
        """Compute learning rate based on annealing ratio"""
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr

    def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, verbose=False):
        """Perform MCMC sampling"""
        lr = self.get_lr(ratio)
        self.mc_prepare(x0hat, xt, model, operator, measurement, sigma)
        self.prepare_prior_score(x0hat, xt, model, sigma)

        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps, desc="MCMC steps") if verbose else range(self.num_steps)

        for _ in pbar:
            cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma)
            epsilon = torch.randn_like(x)
            x = self.mc_update(x, cur_score, lr, epsilon)

            # Clip values to prevent divergence
            x = torch.clamp(x, -10.0, 10.0)

            # Early stopping with NaN
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("[WARNING] NaN/Inf detected in MCMC sampling, returning x0hat")
                return x0hat.detach()

        return x.detach()

    def sample_RL(self, xt, model, x0hat, operator, measurement, sigma, ratio, verbose=False, prev_state=None, rl_steps=1):
        """
        Richardson-Lucy deconvolution step for data consistency.
        Switches to Gaussian MCMC at low sigma.
        """
        # Adaptive measurement guidance based on noise level
        scale = getattr(operator, 'poisson_scale', 100.0)
        limit = 1.0 / np.sqrt(scale)

        sigma_val = sigma.item() if torch.is_tensor(sigma) else sigma
        if sigma_val <= limit and prev_state is not None:
            # Below the noise floor: switch to Gaussian MCMC using clean synthetic measurement
            with torch.no_grad():
                synthetic_y = operator(prev_state)
            return self.sample(xt, model, x0hat, operator, synthetic_y, sigma, ratio, verbose=verbose)

        eps = 1e-8
        # RL update parameters
        alpha = 0.5  # Damping factor
        lam_reg = 0.001  # Regularization strength
        
        # Map measurement to [0, ∞) for Poisson model
        y = torch.clamp((measurement + 1.0) * 0.5, min=0.0) * 255.0
        # Initialize x in positive domain
        x = torch.clamp((x0hat + 1.0) * 0.5, min=eps) * 255.0

        with torch.no_grad():
            for _ in range(rl_steps):
                x_normalized = (x / 255.0).clamp(0, 1) * 2.0 - 1.0
                Kx = operator(x_normalized)
                Kx = torch.clamp((Kx + 1.0) * 0.5, min=eps) * 255.0
                
                ratio_img = y / Kx
                
                # Back-project ratio
                # Since operator (Gaussian blur) is symmetric, K^T = K
                target = ratio_img * 2.0 - 1.0
                smoothed_target = operator(target)
                mult_factor = (smoothed_target + 1.0) * 0.5
                
                # Use clamping to prevent extreme divergence
                mult_factor_clamped = torch.clamp(mult_factor, min=0.1, max=10.0)
                
                # TV gradient (mimicking DiffRL compute_tv_gradient logic directly without extra fn)
                # Simplistic 1-step TV gradient
                diff_x = torch.zeros_like(x_normalized)
                diff_y = torch.zeros_like(x_normalized)
                diff_x[:, :, :, :-1] = x_normalized[:, :, :, 1:] - x_normalized[:, :, :, :-1]
                diff_y[:, :, :-1, :] = x_normalized[:, :, 1:, :] - x_normalized[:, :, :-1, :]
                
                norm_x = torch.sqrt(diff_x[:, :, :, :-1]**2 + diff_y[:, :, :, :-1]**2 + eps)
                norm_y = torch.sqrt(diff_x[:, :, :-1, :]**2 + diff_y[:, :, :-1, :]**2 + eps)
                
                tv_grad = torch.zeros_like(x_normalized)
                tv_grad[:, :, :, 1:] += diff_x[:, :, :, :-1] / norm_x
                tv_grad[:, :, 1:, :] += diff_y[:, :, :-1, :] / norm_y
                tv_grad[:, :, :, :-1] -= diff_x[:, :, :, :-1] / norm_x
                tv_grad[:, :, :-1, :] -= diff_y[:, :, :-1, :] / norm_y
                
                # Apply multiplicative update with TV regularization (Dey-Sutour: 1 + λ ∇TV)
                x = x * (mult_factor_clamped ** alpha) / (1.0 + lam_reg * tv_grad).clamp(min=0.1)
                
            x0y = (x / 255.0).clamp(0, 1) * 2.0 - 1.0
            
            if verbose:
                self.diag = {
                    'sigma': sigma_val,
                    'x0hat_mean': ((x0hat + 1) / 2).mean().item(),
                    'mult_factor_mean': mult_factor_clamped.mean().item(),
                    'mult_factor_std': mult_factor_clamped.std().item(),
                    'mult_factor_min': mult_factor_clamped.min().item(),
                    'mult_factor_max': mult_factor_clamped.max().item(),
                    'ratio_mean': ratio_img.mean().item(),
                    'ratio_std': ratio_img.std().item(),
                    'x0y_mean': ((x0y + 1) / 2).mean().item(),
                    'delta_L2': (x0y - x0hat).pow(2).mean().sqrt().item()
                }

        return x0y
