"""Modular Adaptive AdamW Suite & Verbatim Heavy-Tailed / Spike-Resistant Literature Baselines.

Implements:
  - ModularAdaptiveAdamW: Unified zero-sync optimizer supporting orthogonal ablations across
    first-moment innovation filters (`identity`, `flattop`, `fuld`, `coord_clip`),
    second-moment outlier gates (`raw`, `cap`, `same_as_m`), and prior- vs. denominator-level
    subspace RMS floors (`rho_prior`, `rho_denom`).
  - ExactTAdamW: Verbatim Student-t AdamW following Ilboudo et al. (2020, arXiv:2003.00179, Algorithm 1).
  - ExactStableAdamW: Verbatim RMS-clipped AdamW following Wortsman et al. (NeurIPS 2023, Algorithm 2).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer

V_CUT = 60.0

# Pole-free (3,4) rational approximations of unit-slope normalized symmetric alpha-stable score
# psi_alpha(z) = -d/dz ln p_alpha(z) / (-d^2/dz^2 ln p_alpha(z)|_{z=0}), spliced continuously at |z|=60
# to exact asymptotic tail (alpha + 1) / (p0 * z). Tuple layout: (P_coeffs, Q_coeffs, tail_num, z_peak).
PAPER_RATIONAL_COEFFS: dict[float, tuple[tuple[float, ...], tuple[float, ...], float, float]] = {
    1.5: (
        (1.2695963915423436, 0.15633459124532706, 0.019522143448507407, 0.00022820346128278138),
        (0.28915823558698084, 0.04951803244852624, 0.007245258992812739, 9.143894860749121e-05),
        2.5,
        1.7185172,
    ),
    1.75: (
        (1.106722776, 0.007812274, 0.003280786, 0.000510322),
        (0.058361089, 0.012713271, 0.000210765, 0.000185844),
        2.75,
        2.1840032,
    ),
}


def exact_peak_location(alp: float) -> float:
    """Return positive argmax z_* of the implemented rational score kernel psi_alpha(z)."""
    if alp == 2.0:
        return float("inf")
    if alp == 1.0:
        return 1.0
    return PAPER_RATIONAL_COEFFS[alp][3]


def eval_psi_unit(z: torch.Tensor, alp: float) -> torch.Tensor:
    """Evaluate unit-slope normalized fractional score psi_alpha(z) = grad G_alpha(z) / grad G_alpha'(0)."""
    if alp == 2.0:
        return z
    if alp == 1.0:
        big = z.abs() > 1.0
        z_small = torch.where(big, torch.zeros_like(z), z)
        z_big = torch.where(big, z, torch.ones_like(z))
        inv_z = 1.0 / z_big
        return torch.where(big, inv_z / (1.0 + inv_z * inv_z), z_small / (1.0 + z_small * z_small))

    P, Q, tail_coef, _ = PAPER_RATIONAL_COEFFS[alp]
    p0 = P[0]
    big = z.abs() > V_CUT
    z_small = torch.where(big, torch.zeros_like(z), z)
    z_big = torch.where(big, z, torch.ones_like(z))
    x = z_small * z_small
    num = ((P[3] * x + P[2]) * x + P[1]) * x + p0
    den = ((((Q[3] * x + Q[2]) * x + Q[1]) * x + Q[0]) * x + 1.0) * p0
    return torch.where(big, (tail_coef / p0) / z_big, (z_small * num) / den)


def eval_flattop_twin(z: torch.Tensor, alp: float) -> torch.Tensor:
    """Evaluate exact pre-peak matched monotone flat-top twin P_alpha(z) = sgn(z) * psi_alpha(min(|z|, z_*))."""
    if alp == 2.0:
        return z
    z_star = exact_peak_location(alp)
    return eval_psi_unit(torch.clamp(z, min=-z_star, max=z_star), alp)


class ModularAdaptiveAdamW(Optimizer):
    """Unified zero-sync AdamW testbed for orthogonal innovation-filtering and subspace-floor ablations."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 2.5e-3,
        betas: tuple[float, float] = (0.9, 0.995),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        m_filter: str = "fuld",
        v_filter: str = "cap",
        alp: float = 1.5,
        tau: float = 3.0,
        kappa: float = 2.0,
        rho_prior: float = 0.15,
        rho_denom: float = 0.0,
        max_grad_norm: float | None = None,
    ):
        if m_filter not in ("identity", "flattop", "fuld", "coord_clip"):
            raise ValueError(f"Unknown m_filter={m_filter}")
        if v_filter not in ("raw", "cap", "same_as_m"):
            raise ValueError(f"Unknown v_filter={v_filter}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            m_filter=m_filter,
            v_filter=v_filter,
            alp=alp,
            tau=tau,
            kappa=kappa,
            rho_prior=rho_prior,
            rho_denom=rho_denom,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        max_norm = self.param_groups[0]["max_grad_norm"]
        if max_norm is not None:
            grads_params = [p for group in self.param_groups for p in group["params"] if p.grad is not None]
            if grads_params:
                torch.nn.utils.clip_grad_norm_(grads_params, max_norm)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            m_filter = group["m_filter"]
            v_filter = group["v_filter"]
            alp = group["alp"]
            tau = group["tau"]
            kappa = group["kappa"]
            rho_p = group["rho_prior"]
            rho_d = group["rho_denom"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                if t == 1 or (m_filter == "identity" and v_filter == "raw"):
                    g_tilde = g
                    g_bar = g
                else:
                    bc1_prev = 1.0 - beta1 ** (t - 1)
                    bc2_prev = 1.0 - beta2 ** (t - 1)
                    m_hat_prev = m / bc1_prev
                    v_hat_prev = v / bc2_prev
                    sqrt_v_prev = torch.sqrt(v_hat_prev)

                    if rho_p > 0.0:
                        rms_prev = torch.sqrt(torch.mean(v_hat_prev))
                        s_prev = torch.maximum(sqrt_v_prev, rho_p * rms_prev) + eps
                    else:
                        s_prev = sqrt_v_prev + eps

                    h = tau * s_prev
                    nonzero_mask = g != 0.0

                    if m_filter == "identity":
                        g_tilde = g
                    elif m_filter == "flattop":
                        z = (g - m_hat_prev) / h
                        g_tilde = torch.where(nonzero_mask, m_hat_prev + h * eval_flattop_twin(z, alp), g)
                    elif m_filter == "fuld":
                        z = (g - m_hat_prev) / h
                        g_tilde = torch.where(nonzero_mask, m_hat_prev + h * eval_psi_unit(z, alp), g)
                    elif m_filter == "coord_clip":
                        g_tilde = torch.clamp(g, min=-h, max=h)

                    if v_filter == "raw":
                        g_bar = g
                    elif v_filter == "same_as_m":
                        g_bar = g_tilde
                    else:
                        # Exact passthrough within |g| <= |M_{t-1}| + kappa * h - 1e-12
                        cap_val = m_hat_prev.abs() + (kappa * h)
                        scale = torch.clamp(cap_val / (g.abs() + 1e-12), max=1.0)
                        g_bar = g * scale

                m.mul_(beta1).add_(g_tilde, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_bar, g_bar, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2
                sqrt_v = torch.sqrt(v_hat)

                if rho_d > 0.0:
                    rms_cur = torch.sqrt(torch.mean(v_hat))
                    sigma = torch.maximum(sqrt_v, rho_d * rms_cur) + eps
                else:
                    sigma = sqrt_v + eps

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m_hat, sigma, value=-lr)

        return loss


class ExactTAdamW(Optimizer):
    """Verbatim Ilboudo et al. (2020, arXiv:2003.00179 Algorithm 1) TAdam with decoupled weight decay.

    Note: When w_t == 1, W_t == W_0 == beta_1 / (1 - beta_1) is a stationary fixed point and TAdam
    reduces exactly to standard AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 2.5e-3,
        betas: tuple[float, float] = (0.9, 0.995),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            w_init = beta1 / (1.0 - beta1)
            decay_w = (2.0 * beta1 - 1.0) / beta1

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["W"] = torch.tensor(w_init, device=p.device, dtype=torch.float32)
                elif "W" not in state:
                    state["W"] = torch.tensor(w_init, device=p.device, dtype=torch.float32)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                W_prev = state["W"]

                d_dim = float(p.numel())
                nu = d_dim
                mahal = torch.sum((g - m).pow(2) / (v + eps))
                w_t = (nu + d_dim) / (nu + mahal)

                denom_w = W_prev + w_t
                m.mul_(W_prev / denom_w).add_(g * (w_t / denom_w))
                state["W"] = decay_w * W_prev + w_t

                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                v_hat = v / bc2

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m / bc1, torch.sqrt(v_hat) + eps, value=-lr)

        return loss


class ExactStableAdamW(Optimizer):
    """Verbatim Wortsman et al. (NeurIPS 2023, Algorithm 2 & Appendix F.2) StableAdamW."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 2.5e-3,
        betas: tuple[float, float] = (0.9, 0.995),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        d: float = 1.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, d=d)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            d = group["d"]
            eps_sq = eps * eps

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                rms_t = torch.sqrt(torch.mean(g.pow(2) / torch.clamp(v_hat, min=eps_sq)))
                eta_t = lr / torch.clamp(rms_t / d, min=1.0)

                upd = m_hat / (torch.sqrt(v_hat) + eps) + wd * p.float()
                p.add_((-eta_t * upd).to(p.dtype))

        return loss


def make_optimizer(
    name: str,
    params: Iterable[torch.Tensor],
    lr: float = 2.5e-3,
    betas: tuple[float, float] = (0.9, 0.995),
    wd: float = 0.01,
) -> Optimizer:
    """Instantiate any named experimental arm or baseline with unified hyperparameters."""
    kw = dict(lr=lr, betas=betas, weight_decay=wd)
    if name == "C01_AdamW":
        return ModularAdaptiveAdamW(params, m_filter="identity", v_filter="raw", rho_prior=0.0, rho_denom=0.0, **kw)
    if name == "B16_GradClip":
        return ModularAdaptiveAdamW(
            params, m_filter="identity", v_filter="raw", rho_prior=0.0, rho_denom=0.0, max_grad_norm=1.0, **kw
        )
    if name == "B17_StableAdamW_Exact":
        return ExactStableAdamW(params, **kw)
    if name == "B18_LargeEps_1e-4":
        return ModularAdaptiveAdamW(
            params, eps=1e-4, m_filter="identity", v_filter="raw", rho_prior=0.0, rho_denom=0.0, **kw
        )
    if name == "N19_CoordClip":
        return ModularAdaptiveAdamW(params, m_filter="coord_clip", v_filter="same_as_m", rho_prior=0.0, rho_denom=0.0, **kw)
    if name == "N20_ExactTAdamW":
        return ExactTAdamW(params, **kw)
    if name == "C05_FlatTop_raw":
        return ModularAdaptiveAdamW(params, m_filter="flattop", v_filter="raw", rho_prior=0.0, rho_denom=0.0, **kw)
    if name == "C09_FULD_raw":
        return ModularAdaptiveAdamW(params, m_filter="fuld", v_filter="raw", rho_prior=0.0, rho_denom=0.0, **kw)
    if name == "C11_FULD_cap_noFloor":
        return ModularAdaptiveAdamW(params, m_filter="fuld", v_filter="cap", rho_prior=0.0, rho_denom=0.0, **kw)
    if name == "T13_FlatTop":
        return ModularAdaptiveAdamW(params, m_filter="flattop", v_filter="cap", rho_prior=0.15, rho_denom=0.0, **kw)
    if name == "T13_AdaFULD_PriorFloor":
        return ModularAdaptiveAdamW(params, m_filter="fuld", v_filter="cap", rho_prior=0.15, rho_denom=0.0, **kw)
    if name == "T13_Plus_MildEps":
        return ModularAdaptiveAdamW(
            params, eps=2e-5, m_filter="fuld", v_filter="cap", rho_prior=0.15, rho_denom=0.0, **kw
        )
    raise ValueError(f"Unknown optimizer arm: {name}")
