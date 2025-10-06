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
        """Compute conditional score function"""
        # Data fitting term
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
        data_term = -data_fitting_grad / self.tau ** 2

        # Connection to noisy observation
        xt_term = (xt - x) / sigma ** 2

        # Prior term
        prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)

        # Combine terms with gradient clipping for stability
        total_score = data_term + xt_term + prior_term

        # Clip gradients to prevent NaN
        total_score = torch.clamp(total_score, -100.0, 100.0)

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
