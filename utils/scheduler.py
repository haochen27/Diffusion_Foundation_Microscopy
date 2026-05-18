"""
Schedulers and Samplers for Posterior Sampling
===============================================
EDM scheduler, ODE solvers (Euler, RK4, Heun), and DAPS algorithm
"""

import torch
import numpy as np
import tqdm
from torchdiffeq import odeint
from .mcmc_sampler import MCMCSampler


class EDMSchedulerForDAPS:
    """EDM-based scheduler for DAPS annealing"""

    def __init__(self, num_steps, sigma_max=10.0, sigma_min=0.002, rho=-7.0):
        self.num_steps = num_steps + 1  # include the initial step (matches DiffRL convention)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho

        # Generate sigma schedule
        rho_inv = 1.0 / rho
        timesteps = np.linspace(0, 1, num_steps)
        sigmas = (sigma_max ** rho_inv + timesteps * (sigma_min ** rho_inv - sigma_max ** rho_inv)) ** rho
        self.sigma_steps = torch.tensor(sigmas, dtype=torch.float32)

    def get_prior_sigma(self):
        """Get prior noise level"""
        return self.sigma_max

    # Methods for compatibility with DiffusionPFODE
    def get_scaling(self, t):
        """s(t) = 1 for EDM"""
        return torch.ones_like(self._tensorize(t))

    def get_sigma(self, t):
        """sigma(t) = t for EDM"""
        return self._tensorize(t)

    def get_scaling_derivative(self, t):
        """s'(t) = 0 for EDM"""
        return torch.zeros_like(self._tensorize(t))

    def get_sigma_derivative(self, t):
        """sigma'(t) = 1 for EDM"""
        return torch.ones_like(self._tensorize(t))

    def get_discrete_time_steps(self, num_steps):
        """Generate discrete timesteps (sigma values)"""
        rho_inv = 1.0 / self.rho
        steps = np.linspace(0, 1, num_steps)
        sigmas = (self.sigma_max ** rho_inv + steps * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)) ** self.rho
        return torch.from_numpy(sigmas).float()

    def _tensorize(self, data):
        """Convert to tensor"""
        if isinstance(data, (int, float)):
            return torch.tensor(data).float()
        if isinstance(data, torch.Tensor):
            return data.float()
        return torch.tensor(data).float()


class DiffusionPFODE:
    """
    Diffusion Probability Flow ODE (PF-ODE) for sampling.
    Matches DiffRL's original implementation — num_steps comes from scheduler.
    """

    def __init__(self, model, scheduler, solver='rk4'):
        self.model = model
        self.scheduler = scheduler
        self.solver = solver

    def derivative(self, xt, t):
        """Compute ODE derivative dx/dt (Eq. 4 in EDM paper)"""
        st = self.scheduler.get_scaling(t)
        dst = self.scheduler.get_scaling_derivative(t)
        sigma_t = self.scheduler.get_sigma(t)
        dsigma_t = self.scheduler.get_sigma_derivative(t)
        return dst / st * xt - st * dsigma_t * sigma_t * self.model.score(xt / st, sigma=sigma_t)

    def sample(self, xT, num_steps=None, return_traj=False):
        """Sample from PF-ODE (reverse process: from noisy to clean)"""
        if num_steps is None:
            num_steps = self.scheduler.num_steps

        shape = xT.shape

        def _derivative_wrapper(t, xt):
            xt = xt.view(*shape)
            deriv = self.derivative(xt, t)
            return deriv.flatten(1)

        time_steps = self.scheduler.get_discrete_time_steps(num_steps).to(xT.device)

        with torch.no_grad():
            x_ode_traj = odeint(
                _derivative_wrapper,
                xT.flatten(1),
                time_steps,
                rtol=1e-3,
                atol=1e-3,
                method=self.solver
            )

        x_ode_traj = x_ode_traj.view(num_steps, *shape)

        if return_traj:
            return x_ode_traj
        else:
            return x_ode_traj[-1]


class DAPS:
    """Decoupled Annealing Posterior Sampling (DiffRL-style two-stage).

    Single annealed DAPS loop with fresh-noise re-injection at every step.
    The two stages are implicit in the MCMC data-consistency step:
      Stage 1 (sigma >= 1/sqrt(poisson_scale) = 1/SNR):
          Richardson-Lucy update against the Poisson measurement.
      Stage 2 (sigma <  1/SNR):
          Gaussian MCMC against a synthetic clean measurement
          K(prev_state), i.e. the operator applied to the most-recent x0hat.
    Both stages are handled inside mcmc_sampler.sample_RL; no pure-ODE
    phase is used. Matches ../DiffRL/sampler.py:DAPS behaviour.
    """

    def __init__(self, annealing_steps=20, mcmc_steps=10, mcmc_lr=0.1,
                 tau=0.01, sigma_max=10.0, sigma_min=0.002,
                 solver='tweedie', num_ode_steps=1, sigma_bar=0.5, rho=-7.0, rl_steps=1):
        self.annealing_scheduler = EDMSchedulerForDAPS(
            num_steps=annealing_steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho
        )
        self.mcmc_sampler = MCMCSampler(
            num_steps=mcmc_steps,
            lr=mcmc_lr,
            tau=tau
        )
        self.solver = solver
        self.num_ode_steps = num_ode_steps
        self.sigma_bar = sigma_bar
        self.rl_steps = rl_steps

        print(f"[DAPS] Initialized with {annealing_steps} annealing steps")
        print(f"[DAPS] MCMC: {mcmc_steps} steps, lr={mcmc_lr}, tau={tau}")
        print(f"[DAPS] x0hat: multi-step Euler ({num_ode_steps}) when sigma < sigma_bar={sigma_bar}, else Tweedie")
        print(f"[DAPS] Two-stage data consistency: RL above 1/SNR, Gaussian MCMC below (handled in sample_RL)")

    def sample(self, model, x_start, operator, measurement, verbose=False, save_trajectory=False):
        xt = x_start
        x_prev = None
        trajectory = [] if save_trajectory else None
        use_poisson = getattr(operator, 'use_poisson', False)

        num_iters = self.annealing_scheduler.num_steps - 1
        pbar = tqdm.trange(num_iters, desc="DAPS") if verbose else range(num_iters)

        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            sigma_val = sigma.item()

            with torch.no_grad():
                if sigma_val < self.sigma_bar and self.num_ode_steps > 1:
                    # Multi-step Euler denoising from current sigma down to sigma_min
                    mini_sigmas = np.exp(np.linspace(
                        np.log(sigma_val),
                        np.log(max(self.annealing_scheduler.sigma_min, 0.002)),
                        self.num_ode_steps + 1
                    )).astype(np.float32)
                    x_ode = xt
                    for k in range(self.num_ode_steps):
                        sk = float(mini_sigmas[k])
                        sk1 = float(mini_sigmas[k + 1])
                        s_t = torch.tensor([sk], device=xt.device, dtype=torch.float32)
                        x0hat = model.tweedie(x_ode, s_t)
                        x_ode = x_ode + (x_ode - x0hat) * ((sk1 - sk) / sk)
                else:
                    x0hat = model.tweedie(xt, sigma)

            ratio = step / num_iters
            if use_poisson:
                x0y = self.mcmc_sampler.sample_RL(
                    xt, model, x0hat, operator, measurement, sigma, ratio,
                    verbose=False, prev_state=x_prev, rl_steps=self.rl_steps
                )
            else:
                x0y = self.mcmc_sampler.sample(
                    xt, model, x0hat, operator, measurement, sigma, ratio, verbose=False
                )
            x_prev = x0hat.clone().detach()

            if save_trajectory:
                trajectory.append({
                    'step': step, 'sigma': sigma_val,
                    'x0hat': x0hat.detach().cpu(),
                    'x0y': x0y.detach().cpu(),
                    'xt': xt.detach().cpu()
                })

            # Forward diffusion: re-noise to next annealing sigma (fresh noise every step)
            if step < num_iters - 1:
                sigma_next = self.annealing_scheduler.sigma_steps[step + 1]
                xt = x0y + torch.randn_like(x0y) * sigma_next
            else:
                xt = x0y

            if verbose and isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix({'sigma': f'{sigma_val:.4f}'})

        if save_trajectory:
            return xt, trajectory
        return xt

    def get_start(self, batch_size, model):
        """Generate initial random sample from prior"""
        device = model.device
        in_shape = model.get_in_shape()
        prior_sigma = self.annealing_scheduler.get_prior_sigma()
        x_start = torch.randn(batch_size, *in_shape, device=device) * prior_sigma
        return x_start
