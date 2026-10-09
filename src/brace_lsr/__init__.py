"""BRACE: differential privacy for LSR dense associative memory."""

import jax

jax.config.update("jax_enable_x64", True)

from brace_lsr.brace import (
    active_mask,
    brace_retrieve,
    clean_trajectory,
    naive_retrieve,
    neighboring_query_gap,
    noise_std,
    paired_retrieve,
    project_ball,
    retrieval_map,
    rho_eps_delta,
    selected_step,
    sensitivity_scale,
    smoothing_gamma,
)

__all__ = [
    "active_mask",
    "brace_retrieve",
    "clean_trajectory",
    "naive_retrieve",
    "neighboring_query_gap",
    "noise_std",
    "paired_retrieve",
    "project_ball",
    "retrieval_map",
    "rho_eps_delta",
    "selected_step",
    "sensitivity_scale",
    "smoothing_gamma",
]
