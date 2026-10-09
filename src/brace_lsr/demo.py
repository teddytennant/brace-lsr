"""Section 6.1 two-dimensional BRACE retrieval demo.

The paper specifies n, d, T, r, epsilon, delta, and q. It does not specify
mixture weights, variances, kappa, or the seed. Those choices are documented
in the README and printed here. Printed distances are for this law, not the
paper's reported figures.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from brace_lsr import paired_retrieve


SEED = 0
N = 256
DIMENSION = 2
HORIZON = 5
RADIUS = 0.8
EPSILON = 128.0
DELTA = 1e-5
KAPPA = 0.01
WEIGHTS = (0.4, 0.4, 0.2)
MODE_STD = 0.15
CENTER_STD = 0.45
MEANS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 0.0))
QUERY = (0.2, 0.2)


def sample_gmm(key, n=N):
    """Three-component mixture: modes at (+1, 0) and (-1, 0), center at 0."""
    weights = jnp.asarray(WEIGHTS, dtype=jnp.float64)
    means = jnp.asarray(MEANS, dtype=jnp.float64)
    stds = jnp.asarray((MODE_STD, MODE_STD, CENTER_STD), dtype=jnp.float64)
    key_c, key_n = jax.random.split(key)
    component = jax.random.choice(key_c, 3, shape=(n,), p=weights)
    noise = jax.random.normal(key_n, (n, DIMENSION), dtype=jnp.float64)
    return means[component] + noise * stds[component, None]


def _fmt_pair(point):
    point = jnp.asarray(point)
    return f"({float(point[0]): .5f}, {float(point[1]): .5f})"


def main():
    key_data, key_noise = jax.random.split(jax.random.PRNGKey(SEED))
    bank = sample_gmm(key_data)
    query = jnp.asarray(QUERY, dtype=jnp.float64)
    paired = paired_retrieve(
        bank,
        query,
        RADIUS,
        HORIZON,
        KAPPA,
        EPSILON,
        DELTA,
        key=key_noise,
    )
    brace = paired["brace"]
    naive = paired["naive"]
    clean = paired["clean"]
    fired = [t for t in range(1, HORIZON + 1) if int(brace["correction"][t - 1])]

    print("BRACE selected retrieval, section 6.1 Gaussian mixture")
    print(
        f"n={N} d={DIMENSION} T={HORIZON} r={RADIUS} epsilon={EPSILON} "
        f"delta={DELTA} kappa={KAPPA} seed={SEED}"
    )
    print(
        "weights=0.4,0.4,0.2 mode_std=0.15 center_std=0.45 "
        "center=(0, 0) q=(0.2, 0.2)"
    )
    print(
        f"rho={brace['rho']:.6f} gamma={brace['gamma']:.6f} "
        f"threshold={brace['threshold']:.6f}"
    )
    print(
        "step  clean  naive  brace  corr  residual  "
        "naive_to_clean  brace_to_clean"
    )
    for t in range(1, HORIZON + 1):
        naive_dist = float(jnp.linalg.norm(naive["Y"][t] - clean[t]))
        brace_dist = float(jnp.linalg.norm(brace["Y"][t] - clean[t]))
        print(
            f"{t:4d}  {_fmt_pair(clean[t])}  {_fmt_pair(naive['Y'][t])}  "
            f"{_fmt_pair(brace['Y'][t])}  {int(brace['correction'][t - 1])}  "
            f"{float(brace['residual_norm'][t - 1]):.5f}  "
            f"{naive_dist:.5f}  {brace_dist:.5f}"
        )
    if fired:
        print("correction fired at steps: " + ", ".join(str(t) for t in fired))
    else:
        print("correction fired at steps: none")
    print("naive and BRACE used the same xi")


if __name__ == "__main__":
    main()
