"""Tests for selected BRACE retrieval."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from brace_lsr import (
    active_mask,
    brace_retrieve,
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


def test_empty_active_set_returns_state():
    bank = jnp.array([[3.0, 0.0], [-4.0, 1.0], [0.0, 2.5]])
    state = jnp.array([0.0, 0.0])
    mapped, count, mask = retrieval_map(bank, state, 1.0)
    assert int(count) == 0
    assert not bool(jnp.any(mask))
    assert np.allclose(mapped, state)


def test_nonempty_retrieval_is_mean_and_inside_ball():
    state = jnp.array([0.0, 0.0])
    radius = 1.0
    bank = jnp.array(
        [
            [0.6, 0.0],
            [0.0, 0.8],
            [1.0, 0.0],
            [1.1, 0.0],
        ]
    )
    mapped, count, mask = retrieval_map(bank, state, radius)
    assert int(count) == 3
    assert np.array_equal(np.asarray(mask), np.array([True, True, True, False]))
    expected = np.array([1.6 / 3.0, 0.8 / 3.0])
    assert np.allclose(mapped, expected)
    assert np.linalg.norm(np.asarray(mapped) - np.asarray(state)) <= radius + 1e-12

    key = jax.random.PRNGKey(2)
    random_bank = jax.random.normal(key, (12, 3))
    query = jnp.array([0.2, -0.4, 0.1])
    mapped_r, count_r, _ = retrieval_map(random_bank, query, 1.5)
    assert int(count_r) > 0
    assert float(jnp.linalg.norm(mapped_r - query)) <= 1.5 + 1e-12


def test_rho_identity():
    pairs = [
        (128.0, 1e-5),
        (1.0, 1e-5),
        (0.5, 1e-6),
        (10.0, 1e-3),
        (2.0, 1e-8),
    ]
    for epsilon, delta in pairs:
        rho = float(rho_eps_delta(epsilon, delta))
        log_inv = math.log(1.0 / delta)
        recovered = rho + 2.0 * math.sqrt(rho * log_inv)
        assert abs(recovered - epsilon) < 1e-8


def test_ball_projection():
    center = jnp.array([1.0, -1.0])
    assert np.allclose(project_ball(center, center, 0.5), center)
    inside = jnp.array([1.2, -1.1])
    assert np.allclose(project_ball(inside, center, 1.0), inside)
    offset = jnp.array([3.0, 4.0])
    projected = project_ball(center + offset, center, 2.0)
    assert np.allclose(projected, center + offset * 0.4)
    assert abs(float(jnp.linalg.norm(projected - center)) - 2.0) < 1e-12


def test_correction_logic_no_noise():
    bank = jnp.array([[0.0, 0.0], [1.5, 0.0]])
    radius = 1.0
    horizon = 2
    kappa = 0.01
    z_prev = jnp.array([0.0, 0.0])
    y_prev = jnp.array([0.6, 0.0])
    step = selected_step(bank, z_prev, y_prev, radius, horizon, kappa)
    assert step["correction"]
    assert float(jnp.linalg.norm(step["z"] - y_prev)) <= radius
    assert np.allclose(step["query"], step["z"])
    assert not np.allclose(step["query"], step["a"])

    bank_b = jnp.array([[0.0, 0.0], [0.5, 0.0]])
    z_b = jnp.array([0.0, 0.0])
    y_b = jnp.array([-0.6, 0.0])
    # ||R|| = 0.25, threshold = 0.3 * 1 * 1 = 0.3, so the correction stays off.
    step_b = selected_step(bank_b, z_b, y_b, radius, 1, 0.3)
    assert not step_b["correction"]
    assert float(step_b["residual_norm"]) < step_b["threshold"]
    assert np.allclose(step_b["query"], project_ball(step_b["a"], y_b, radius))
    assert not np.allclose(step_b["a"], step_b["z"])


def test_released_trajectory_stays_in_ball():
    key = jax.random.PRNGKey(7)
    k_bank, k_noise = jax.random.split(key)
    bank = jax.random.normal(k_bank, (16, 3), dtype=jnp.float64)
    query = jnp.array([0.3, -0.2, 0.1])
    radius = 0.6
    horizon = 4
    paired = paired_retrieve(
        bank, query, radius, horizon, 0.02, 1.0, 1e-5, key=k_noise
    )
    for name in ("brace", "naive"):
        released = paired[name]["Y"]
        for t in range(1, horizon + 1):
            distance = float(jnp.linalg.norm(released[t] - query))
            assert distance <= (t + 1) * radius + 1e-8
    assert np.allclose(paired["brace"]["xi"], paired["naive"]["xi"])


def test_correction_keeps_clean_update_under_shared_noise():
    bank = jnp.array([[0.0, 0.0], [1.5, 0.0]])
    query = jnp.array([0.0, 0.0])
    radius = 1.0
    horizon = 2
    kappa = 0.01
    noise = jnp.array([[0.6, 0.0], [0.0, 0.0]])
    brace = brace_retrieve(
        bank, query, radius, horizon, kappa, 1.0, 1e-5, noise=noise
    )
    naive = naive_retrieve(
        bank, query, radius, horizon, kappa, 1.0, 1e-5, noise=noise
    )
    assert np.allclose(brace["xi"], noise)
    assert np.allclose(naive["xi"], noise)
    assert np.allclose(brace["Y"][1], jnp.array([0.6, 0.0]))

    clean_mask = active_mask(bank, brace["z"][1], radius)
    naive_mask = active_mask(bank, brace["Y"][1], radius)
    assert not np.array_equal(np.asarray(clean_mask), np.asarray(naive_mask))

    # Step t=2 is index 1. Projection of z_2 onto B(Y_1, r) does not bind.
    assert float(jnp.linalg.norm(brace["z"][2] - brace["Y"][1])) <= radius
    assert np.allclose(brace["Q"][1], brace["z"][2])
    assert not np.allclose(naive["Q"][1], brace["Q"][1])
    assert int(brace["correction"][1]) == 1
    assert int(naive["correction"][1]) == 0


def test_neighbor_certificate():
    key = jax.random.PRNGKey(11)
    k_bank, k_noise, k_repl = jax.random.split(key, 3)
    n = 8
    dimension = 2
    radius = 0.8
    horizon = 4
    kappa = 0.01
    epsilon = 1.0
    delta = 1e-5
    bank = jax.random.normal(k_bank, (n, dimension), dtype=jnp.float64)
    query = jnp.array([0.1, -0.15])
    run = brace_retrieve(
        bank, query, radius, horizon, kappa, epsilon, delta, key=k_noise
    )
    repl_keys = jax.random.split(k_repl, 20)
    checked = 0
    for rk in repl_keys:
        k_idx, k_val = jax.random.split(rk)
        index = int(jax.random.randint(k_idx, (), 0, n))
        value = jax.random.normal(k_val, (dimension,), dtype=jnp.float64) * 2.0
        prime = bank.at[index].set(value)
        for step in range(1, horizon + 1):
            cert = neighboring_query_gap(
                bank,
                prime,
                query,
                radius,
                horizon,
                kappa,
                epsilon,
                delta,
                step,
                run["Y"],
            )
            assert cert["holds"], (step, cert["gap"], cert["delta_t"])
            assert cert["gap"] <= cert["delta_t"] + 1e-8
            checked += 1
        mask = active_mask(bank, query, radius)
        count = int(jnp.sum(mask))
        local = 2.0 * radius / max(1, count)
        step1 = neighboring_query_gap(
            bank,
            prime,
            query,
            radius,
            horizon,
            kappa,
            epsilon,
            delta,
            1,
            run["Y"],
        )
        assert step1["gap"] <= local + 1e-8
    assert checked >= 20


def test_gamma_inequalities():
    cases = [
        (128.0, 1e-5, 5, 2),
        (1.0, 1e-5, 5, 2),
        (1.0, 1e-5, 1, 1),
        (10.0, 1e-3, 8, 4),
    ]
    for epsilon, delta, horizon, dimension in cases:
        gamma = float(smoothing_gamma(epsilon, delta, horizon, dimension))
        rho = float(rho_eps_delta(epsilon, delta))
        alpha = 1.0 + math.sqrt(math.log(1.0 / delta) / rho)
        assert gamma > 0
        assert gamma <= 0.25 + 1e-12
        assert gamma <= 1.0 / (8.0 * (alpha - 1.0)) + 1e-12
        numer = alpha * math.log(alpha) - (alpha - 1.0) * math.log(alpha - 1.0)
        denom = (alpha - 1.0) * (
            8.0 * alpha * (alpha - 1.0) * rho + 5.0 * horizon * dimension
        )
        third = numer / denom
        assert math.isfinite(third) and third > 0
        assert gamma <= third + 1e-12
        expected = min(0.25, 1.0 / (8.0 * (alpha - 1.0)), third)
        assert abs(gamma - expected) < 1e-12


def test_sensitivity_scale_matches_envelope():
    radius = 0.8
    n = 8
    gamma = 0.1
    local_off = 2.0 * radius / 4
    smooth_off = max(local_off, 2.0 * radius * math.exp(-gamma))
    assert abs(
        sensitivity_scale(False, radius, n, 4, gamma)
        - max(2.0 * radius / n, smooth_off)
    ) < 1e-12
    local_on = 2.0 * radius
    smooth_on = max(local_on, 2.0 * radius * math.exp(-gamma))
    assert abs(
        sensitivity_scale(True, radius, n, 4, gamma)
        - max(2.0 * radius / n, smooth_on)
    ) < 1e-12
    assert abs(noise_std(0.4, 5, 2.0) - 0.4 * math.sqrt(5.0 / 4.0)) < 1e-12
