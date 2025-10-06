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

    def __init__(self, num_steps, sigma_max=10.0, sigma_min=0.002, rho=7.0):
        self.num_steps = num_steps
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

    Supports multiple solvers:
    - 'tweedie': Direct denoising (1-step, order 0)
    - 'euler': Euler method (order 1)
    - 'rk4': 4th order Runge-Kutta
    - 'dopri5': Dormand-Prince adaptive method
    """

    def __init__(self, model, scheduler, solver='tweedie', num_steps=50):
        """
        Args:
            model: Diffusion model with score() method
            scheduler: Scheduler with scaling/sigma functions
            solver: ODE solver type ('tweedie', 'euler', 'rk4', 'dopri5', 'heun')
            num_steps: Number of integration steps (ignored for adaptive solvers)
        """
        self.model = model
        self.scheduler = scheduler
        self.solver = solver
        self.num_steps = num_steps if num_steps is not None else scheduler.num_steps

    def derivative(self, xt, t):
        """
        Compute ODE derivative dx/dt

        Following EDM paper Eq. (4):
        dx/dt = [s'(t)/s(t)] * x - s(t) * sigma'(t) * sigma(t) * score(x/s(t), sigma(t))

        For EDM: s(t) = 1, sigma(t) = t
        Simplifies to: dx/dt = -sigma(t) * score(x, sigma(t))
        """
        st = self.scheduler.get_scaling(t)
        dst = self.scheduler.get_scaling_derivative(t)
        sigma_t = self.scheduler.get_sigma(t)
        dsigma_t = self.scheduler.get_sigma_derivative(t)

        return dst / st * xt - st * dsigma_t * sigma_t * self.model.score(xt / st, sigma=sigma_t)

    def sample(self, xT, return_traj=False):
        """
        Sample from PF-ODE (reverse process: from noisy to clean)

        Args:
            xT: Noisy input tensor
            return_traj: Whether to return full trajectory

        Returns:
            Denoised tensor (or trajectory if return_traj=True)
        """
        if self.solver == 'tweedie':
            # Direct denoising using Tweedie's formula (0-step, just use the model)
            sigma = self.scheduler.get_sigma(self.scheduler.sigma_steps[0])
            with torch.no_grad():
                x0hat = self.model.tweedie(xT, sigma)
            return x0hat

        # Higher-order ODE solvers using torchdiffeq
        shape = xT.shape

        def _derivative_wrapper(t, xt):
            xt = xt.view(*shape)
            deriv = self.derivative(xt, t)
            return deriv.flatten(1)

        # Get time steps (sigma values)
        time_steps = self.scheduler.get_discrete_time_steps(self.num_steps).to(xT.device)

        # Solve ODE: from high sigma to low sigma
        with torch.no_grad():
            x_ode_traj = odeint(
                _derivative_wrapper,
                xT.flatten(1),
                time_steps,
                rtol=1e-3,
                atol=1e-3,
                method=self.solver
            )  # [num_steps, B, D]

        x_ode_traj = x_ode_traj.view(self.num_steps, *shape)

        if return_traj:
            return x_ode_traj
        else:
            return x_ode_traj[-1]


class DAPS:
    """Decoupled Annealing Posterior Sampling"""

    def __init__(self, annealing_steps=20, mcmc_steps=10, mcmc_lr=0.1,
                 tau=0.01, sigma_max=10.0, sigma_min=0.002,
                 solver='tweedie', num_ode_steps=50):
        """
        Args:
            annealing_steps: Number of annealing levels
            mcmc_steps: MCMC steps per annealing level
            mcmc_lr: MCMC learning rate
            tau: Temperature for data fitting
            sigma_max: Maximum noise level
            sigma_min: Minimum noise level
            solver: ODE solver type ('tweedie', 'euler', 'rk4', 'dopri5', 'heun')
            num_ode_steps: Number of ODE integration steps
        """
        self.annealing_scheduler = EDMSchedulerForDAPS(
            num_steps=annealing_steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min
        )
        self.mcmc_sampler = MCMCSampler(
            num_steps=mcmc_steps,
            lr=mcmc_lr,
            tau=tau
        )
        self.solver = solver
        self.num_ode_steps = num_ode_steps

        print(f"[DAPS] Initialized with {annealing_steps} annealing steps")
        print(f"[DAPS] MCMC: {mcmc_steps} steps, lr={mcmc_lr}, tau={tau}")
        print(f"[DAPS] ODE Solver: {solver}" + (f" with {num_ode_steps} steps" if solver != 'tweedie' else " (direct denoising)"))

    def sample(self, model, x_start, operator, measurement, verbose=False, save_trajectory=False):
        """
        Perform DAPS posterior sampling

        Args:
            model: Diffusion model
            x_start: Initial noise sample
            operator: Forward operator (e.g., Gaussian blur)
            measurement: Observed measurement
            verbose: Show progress bar
            save_trajectory: Whether to save intermediate steps

        Returns:
            Final sampled image (or tuple of (final, trajectory) if save_trajectory=True)
        """
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1, desc="DAPS") if verbose else \
               range(self.annealing_scheduler.num_steps - 1)

        xt = x_start
        trajectory = [] if save_trajectory else None

        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]

            # 1. Reverse diffusion (PF-ODE)
            with torch.no_grad():
                if self.solver == 'tweedie':
                    # Direct denoising for order 0
                    x0hat = model.tweedie(xt, sigma)
                else:
                    # Use ODE solver for higher order
                    # Create a scheduler instance at current sigma level
                    pfode_scheduler = EDMSchedulerForDAPS(
                        num_steps=self.num_ode_steps,
                        sigma_max=sigma.item() if torch.is_tensor(sigma) else sigma,
                        sigma_min=0.002
                    )
                    pfode_sampler = DiffusionPFODE(
                        model,
                        pfode_scheduler,
                        solver=self.solver,
                        num_steps=self.num_ode_steps
                    )
                    x0hat = pfode_sampler.sample(xt)

            # 2. MCMC update for data consistency
            ratio = step / (self.annealing_scheduler.num_steps - 1)
            x0y = self.mcmc_sampler.sample(
                xt, model, x0hat, operator, measurement, sigma, ratio, verbose=False
            )

            # Save intermediate results
            if save_trajectory:
                trajectory.append({
                    'step': step,
                    'sigma': sigma.item() if torch.is_tensor(sigma) else sigma,
                    'x0hat': x0hat.detach().cpu(),
                    'x0y': x0y.detach().cpu(),
                    'xt': xt.detach().cpu()
                })

            # 3. Forward diffusion to next noise level
            if step != self.annealing_scheduler.num_steps - 2:
                sigma_next = self.annealing_scheduler.sigma_steps[step + 1]
                xt = x0y + torch.randn_like(x0y) * sigma_next
            else:
                xt = x0y

            if verbose and isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix({'sigma': f'{sigma:.4f}'})

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
