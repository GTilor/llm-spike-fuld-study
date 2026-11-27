"""Modular adaptive optimizers and exact literature baselines for LLM spike & domain-shift studies."""

from .optimizers import (
    PAPER_RATIONAL_COEFFS,
    V_CUT,
    ExactStableAdamW,
    ExactTAdamW,
    ModularAdaptiveAdamW,
    eval_flattop_twin,
    eval_psi_unit,
    exact_peak_location,
    make_optimizer,
)

__all__ = [
    "V_CUT",
    "PAPER_RATIONAL_COEFFS",
    "exact_peak_location",
    "eval_psi_unit",
    "eval_flattop_twin",
    "ModularAdaptiveAdamW",
    "ExactTAdamW",
    "ExactStableAdamW",
    "make_optimizer",
]
