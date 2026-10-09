"""Selected BRACE retrieval (Algorithm 1 of arXiv 2610.11218).

The released map is the selected rule in the algorithm box: clean LSR
retrieval stays internal, a thresholded correction is projected into the
ball of radius r about the previous release, and isotropic Gaussian noise
is added at the smooth-sensitivity scale.
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Absolute slack for the neighboring-bank check. This covers float64
# roundoff in the norm, not a relaxation of the sensitivity bound.
_CERT_ATOL = 1e-8


def project_ball(point, center, radius):
    """Euclidean projection onto the closed ball B(center, radius).

    Pi(c, rad)(x) = c + (x - c) * min(1, rad / ||x - c||), and the identity
    when x is c.
    """
    point = jnp.asarray(point, dtype=jnp.float64)
    center = jnp.asarray(center, dtype=jnp.float64)
    radius = jnp.asarray(radius, dtype=jnp.float64)
    delta = point - center
    norm = jnp.linalg.norm(delta)
    safe_norm = jnp.where(norm > 0, norm, 1.0)
    scale = jnp.where(norm > 0, jnp.minimum(1.0, radius / safe_norm), 1.0)
    return center + delta * scale


def active_mask(bank, state, radius):
    """I(D, state) = {i : ||X_i - state||_2 <= radius}, as a boolean mask."""
    bank = jnp.asarray(bank, dtype=jnp.float64)
    state = jnp.asarray(state, dtype=jnp.float64)
    delta = bank - state
    dist2 = jnp.sum(delta * delta, axis=-1)
    return dist2 <= float(radius) ** 2


def retrieval_map(bank, state, radius):
    """LSR retrieval Phi_D(state).

    Returns the state itself when the active set is empty, otherwise the
    mean of the memory points inside the closed ball of radius r.
    Also returns the active count and the boolean mask.
    """
    bank = jnp.asarray(bank, dtype=jnp.float64)
    state = jnp.asarray(state, dtype=jnp.float64)
    mask = active_mask(bank, state, radius)
    count = jnp.sum(mask)
    total = jnp.sum(bank * mask[:, None], axis=0)
    mean = total / jnp.maximum(count, 1.0)
    mapped = jnp.where(count > 0, mean, state)
    return mapped, count, mask


def rho_eps_delta(epsilon, delta):
    """zCDP rho solving epsilon = rho + 2 sqrt(rho log(1/delta)).

    The PDF typesetting places the square roots so that they cancel and rho
    collapses to epsilon squared. The consistent reading of the zCDP
    conversion, and of alpha = 1 + sqrt(log(1/delta) / rho) in appendix C.2,
    is the positive root

        rho = (sqrt(log(1/delta) + epsilon) - sqrt(log(1/delta)))^2.
    """
    epsilon = jnp.asarray(epsilon, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    log_inv = jnp.log(1.0 / delta)
    return (jnp.sqrt(log_inv + epsilon) - jnp.sqrt(log_inv)) ** 2


def smoothing_gamma(epsilon, delta, horizon, dimension):
    """Upper bound on gamma_t from equation (8).

    The paper allows any smaller positive gamma_t. This returns the upper
    bound itself. If the third term is non-positive or non-finite, it is
    dropped and the bound is min(1/4, 1/(8 (alpha - 1))).
    """
    rho = rho_eps_delta(epsilon, delta)
    log_inv = jnp.log(1.0 / jnp.asarray(delta, dtype=jnp.float64))
    alpha = 1.0 + jnp.sqrt(log_inv / rho)
    cap_quarter = jnp.asarray(0.25, dtype=jnp.float64)
    cap_alpha = 1.0 / (8.0 * (alpha - 1.0))
    numer = alpha * jnp.log(alpha) - (alpha - 1.0) * jnp.log(alpha - 1.0)
    denom = (alpha - 1.0) * (
        8.0 * alpha * (alpha - 1.0) * rho
        + 5.0 * float(horizon) * float(dimension)
    )
    third = numer / denom
    gamma = jnp.minimum(cap_quarter, cap_alpha)
    third_ok = jnp.isfinite(third) & (third > 0)
    return jnp.where(third_ok, jnp.minimum(gamma, third), gamma)


def sensitivity_scale(correction_on, radius, n, active_count, gamma):
    """Delta_t = max(2r/n, S) with S = max(L0, 2r exp(-gamma)).

    L0 = 2r / max(1, |J|) when the correction indicator is off, else L0 = 2r.
    The paper states L_t <= 2r always. S dominates max_{k >= 1} exp(-gamma k)
    * 2r, and equals the smooth sensitivity of an envelope that is L0 at the
    current bank and never exceeds 2r. This replaces CorrectedCountSearch.
    """
    radius = float(radius)
    n = int(n)
    active_count = int(active_count)
    gamma = float(gamma)
    if correction_on:
        local = 2.0 * radius
    else:
        local = 2.0 * radius / max(1, active_count)
    smooth = max(local, 2.0 * radius * math.exp(-gamma))
    return max(2.0 * radius / n, smooth)


def noise_std(delta_t, horizon, rho):
    """Coordinate std of xi_t ~ N(0, T Delta_t^2 / (2 rho) I_d)."""
    return float(delta_t) * math.sqrt(float(horizon) / (2.0 * float(rho)))


def clean_trajectory(bank, query, radius, horizon):
    """Iterate Phi from the query. Index t of the result is z_t, with z_0 = q."""
    bank = jnp.asarray(bank, dtype=jnp.float64)
    state = jnp.asarray(query, dtype=jnp.float64)
    states = [state]
    for _ in range(int(horizon)):
        state, _, _ = retrieval_map(bank, state, radius)
        states.append(state)
    return jnp.stack(states)


def selected_step(bank, z_prev, y_prev, radius, horizon, kappa):
    """One selected-rule step before noise.

    R_t = z_t - a_t, which the appendix shows equals the symmetric-difference
    formula in equation (1). H_t = 1{||R_t||_2 > kappa T r^2}. Q_t is the
    projection of a_t + H_t R_t onto B(Y_{t-1}, r).
    """
    z_t, i_count, _ = retrieval_map(bank, z_prev, radius)
    a_t, j_count, _ = retrieval_map(bank, y_prev, radius)
    residual = z_t - a_t
    residual_norm = float(jnp.linalg.norm(residual))
    threshold = float(kappa) * float(horizon) * float(radius) ** 2
    correction = residual_norm > threshold
    corrected = a_t + (1.0 if correction else 0.0) * residual
    query = project_ball(corrected, y_prev, radius)
    return {
        "z": z_t,
        "a": a_t,
        "residual": residual,
        "residual_norm": residual_norm,
        "threshold": threshold,
        "correction": correction,
        "query": query,
        "active_clean": int(i_count),
        "active_released": int(j_count),
    }


def _as_bank_query(bank, query):
    bank = jnp.asarray(bank, dtype=jnp.float64)
    query = jnp.asarray(query, dtype=jnp.float64)
    if bank.ndim != 2:
        raise ValueError("bank must have shape (n, d)")
    if query.shape != (bank.shape[1],):
        raise ValueError("query must have shape (d,)")
    return bank, query


def _run(bank, query, radius, horizon, kappa, epsilon, delta, noise, key, correct):
    bank, query = _as_bank_query(bank, query)
    n, dimension = bank.shape
    horizon = int(horizon)
    rho = float(rho_eps_delta(epsilon, delta))
    gamma = float(smoothing_gamma(epsilon, delta, horizon, dimension))
    if noise is None:
        if key is None:
            raise ValueError("provide key or noise")
        innovations = jax.random.normal(
            key, (horizon, dimension), dtype=jnp.float64
        )
    else:
        noise = jnp.asarray(noise, dtype=jnp.float64)
        if noise.shape != (horizon, dimension):
            raise ValueError("noise must have shape (T, d)")
        innovations = None

    z = query
    y = query
    zs = [z]
    ys = [y]
    queries = []
    anchors = []
    corrections = []
    residual_norms = []
    deltas = []
    stds = []
    xis = []
    for t in range(1, horizon + 1):
        step = selected_step(bank, z, y, radius, horizon, kappa)
        if not correct:
            step = dict(step)
            step["correction"] = False
            step["query"] = project_ball(step["a"], y, radius)
        delta_t = sensitivity_scale(
            step["correction"],
            radius,
            n,
            step["active_released"],
            gamma,
        )
        std = noise_std(delta_t, horizon, rho)
        if innovations is None:
            xi = noise[t - 1]
        else:
            xi = innovations[t - 1] * std
        y_tilde = step["query"] + xi
        y = project_ball(y_tilde, query, (t + 1) * float(radius))
        z = step["z"]
        zs.append(z)
        ys.append(y)
        queries.append(step["query"])
        anchors.append(step["a"])
        corrections.append(1 if step["correction"] else 0)
        residual_norms.append(step["residual_norm"])
        deltas.append(delta_t)
        stds.append(std)
        xis.append(xi)
    return {
        "z": jnp.stack(zs),
        "Y": jnp.stack(ys),
        "Q": jnp.stack(queries),
        "a": jnp.stack(anchors),
        "correction": jnp.asarray(corrections, dtype=jnp.int32),
        "residual_norm": jnp.asarray(residual_norms, dtype=jnp.float64),
        "delta": jnp.asarray(deltas, dtype=jnp.float64),
        "std": jnp.asarray(stds, dtype=jnp.float64),
        "xi": jnp.stack(xis),
        "rho": rho,
        "gamma": gamma,
        "threshold": float(kappa) * float(horizon) * float(radius) ** 2,
    }


def brace_retrieve(
    bank,
    query,
    radius,
    horizon,
    kappa,
    epsilon,
    delta,
    key=None,
    noise=None,
):
    """Selected BRACE retrieval.

    z has shape (T+1, d) with z_0 = q. Y has the same shape, Y_0 = q.
    Q, a, correction, residual_norm, delta, std, and xi are indexed by
    step t = 1..T at position t-1. If noise is given it is used as xi, already
    scaled. Otherwise key draws standard normals and scales them by
    Delta_t * sqrt(T / (2 rho)).
    """
    return _run(
        bank, query, radius, horizon, kappa, epsilon, delta, noise, key, True
    )


def naive_retrieve(
    bank,
    query,
    radius,
    horizon,
    kappa,
    epsilon,
    delta,
    key=None,
    noise=None,
):
    """Same recursion as BRACE, but Q_t is the projection of a_t with no correction.

    Pass the xi array from brace_retrieve to share the noise scale and the
    same Gaussian draw, as in the section 6.1 comparison.
    """
    return _run(
        bank, query, radius, horizon, kappa, epsilon, delta, noise, key, False
    )


def paired_retrieve(
    bank,
    query,
    radius,
    horizon,
    kappa,
    epsilon,
    delta,
    key=None,
    noise=None,
):
    """Clean, naive, and BRACE trajectories with one shared xi sequence.

    The scale and the draw come from the BRACE path. Naive adds that same xi
    to its uncorrected query.
    """
    brace = brace_retrieve(
        bank,
        query,
        radius,
        horizon,
        kappa,
        epsilon,
        delta,
        key=key,
        noise=noise,
    )
    naive = naive_retrieve(
        bank,
        query,
        radius,
        horizon,
        kappa,
        epsilon,
        delta,
        noise=brace["xi"],
    )
    return {"clean": brace["z"], "naive": naive, "brace": brace}


def neighboring_query_gap(
    bank,
    bank_prime,
    query,
    radius,
    horizon,
    kappa,
    epsilon,
    delta,
    step,
    y_history,
):
    """||Q_t(D) - Q_t(D')|| for one point replacement, and whether it is <= Delta_t.

    The released history is fixed. Each bank recomputes its clean trajectory
    from q, matching the paper's neighboring-bank comparison at a fixed y_<t.
    step is 1-based. y_history[k] is Y_k, with Y_0 = q.
    """
    bank, query = _as_bank_query(bank, query)
    bank_prime = jnp.asarray(bank_prime, dtype=jnp.float64)
    if bank_prime.shape != bank.shape:
        raise ValueError("neighbor bank must have the same shape")
    y_history = jnp.asarray(y_history, dtype=jnp.float64)
    step = int(step)
    if step < 1 or step > int(horizon):
        raise ValueError("step must lie in 1..T")
    if y_history.shape[0] < step:
        raise ValueError("y_history must include Y_{t-1}")

    def clean_before(memory, steps):
        state = query
        for _ in range(steps):
            state, _, _ = retrieval_map(memory, state, radius)
        return state

    y_prev = y_history[step - 1]
    step_d = selected_step(
        bank,
        clean_before(bank, step - 1),
        y_prev,
        radius,
        horizon,
        kappa,
    )
    step_p = selected_step(
        bank_prime,
        clean_before(bank_prime, step - 1),
        y_prev,
        radius,
        horizon,
        kappa,
    )
    gap = float(jnp.linalg.norm(step_d["query"] - step_p["query"]))
    n, dimension = bank.shape
    gamma = float(smoothing_gamma(epsilon, delta, horizon, dimension))
    local = (
        2.0 * float(radius)
        if step_d["correction"]
        else 2.0 * float(radius) / max(1, step_d["active_released"])
    )
    smooth = max(local, 2.0 * float(radius) * math.exp(-gamma))
    delta_t = sensitivity_scale(
        step_d["correction"],
        radius,
        n,
        step_d["active_released"],
        gamma,
    )
    return {
        "gap": gap,
        "delta_t": delta_t,
        "holds": gap <= delta_t + _CERT_ATOL,
        "local": local,
        "smooth": smooth,
        "correction": step_d["correction"],
        "query": step_d["query"],
        "query_prime": step_p["query"],
    }
