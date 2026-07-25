"""
clynx.optim.mulion
~~~~~~~~~~~~~~~~~~
Mulion — unified optimizer for Transformer training.

Automatically routes parameters by dimensionality:
- 2D+ params (weight matrices) → Muon  (Newton-Schulz orthogonalization)
- 1D params  (bias, scale)     → Lion  (sign of momentum only)

Both optimizers are magnitude-agnostic, making them naturally compatible
with the symlog-compressed gradients from clynx.slam.functional.symlog_bwd.

Memory footprint (bfloat16 params, fp32 optimizer state):
    params       : 2 bytes
    momentum     : 4 bytes  (fp32, 1 buffer — both Muon and Lion)
    ─────────────────────────
    total        : 6 bytes per parameter

Compare to Adam: 2 + 4 (m) + 4 (v) = 10 bytes per parameter.

Setting ns_steps=0 degrades Muon → SGD momentum (no orthogonalization),
useful for ablation studies.

No external optimizer library dependency — pure JAX.
Interface mirrors optax.GradientTransformation: (init_fn, update_fn) pair.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple, Any, Callable, Tuple


# ---------------------------------------------------------------------------
# GradientTransformation — minimal optax-compatible interface (no optax dep)
# ---------------------------------------------------------------------------

class GradientTransformation(NamedTuple):
    """
    Minimal optimizer interface, mirrors optax.GradientTransformation.

        init   : params -> state
        update : (grads, state, params=None) -> (updates, new_state)

    Drop-in compatible: if you later add optax, this NamedTuple unpacks
    into the same (init, update) protocol.
    """
    init: Callable
    update: Callable


# Register as a static pytree node (no leaves): the two callables are
# aux data, not array leaves. Without this, embedding a
# GradientTransformation inside a jit-traced pytree (e.g. TrainState.opt)
# makes JAX try to treat init_fn/update_fn as abstract array leaves and
# fail with "Error interpreting argument ... as an abstract array."
jax.tree_util.register_pytree_node(
    GradientTransformation,
    lambda gt: ((), gt),          # no leaves; the whole namedtuple is aux data
    lambda gt, _: gt,              # reconstruct as-is
)


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

def newton_schulz(G: Array, steps: int = 5) -> Array:
    """
    Iterative Newton-Schulz to approximate orthogonal factor of G.

    Converges to the orthogonal matrix in the polar decomposition of G.
    After convergence, singular values ≈ 1 — magnitude is discarded,
    only the directional structure of the gradient matrix is retained.

    steps=0 → returns G unchanged (degrades to SGD momentum).
    """
    if steps == 0:
        return G

    # Coefficients for the quintic iteration (Kosson et al.)
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Normalize to avoid numerical issues
    G = G / (jnp.linalg.norm(G) + 1e-7)

    # Ensure G is tall (rows >= cols) for stability
    transposed = G.shape[0] < G.shape[1]
    if transposed:
        G = G.T

    for _ in range(steps):
        A = G @ G.T
        G = a * G + b * (A @ G) + c * (A @ A @ G)

    if transposed:
        G = G.T

    return G


# ---------------------------------------------------------------------------
# Optimizer state
# ---------------------------------------------------------------------------

class MulionState(NamedTuple):
    """
    Optimizer state for Mulion.

    A single momentum buffer per parameter, stored in fp32.
    No second moment — keeps memory at 6 bytes/param in bfloat16 training.
    """
    momentum: Any   # pytree matching params, dtype fp32
    count: Array    # step counter, shape ()


# ---------------------------------------------------------------------------
# Mulion optimizer
# ---------------------------------------------------------------------------

def mulion(
    lr: float = 1e-3,
    momentum: float = 0.95,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
) -> GradientTransformation:
    """
    Mulion optimizer.

    Parameters
    ----------
    lr : float
        Learning rate. Applied after orthogonalization (Muon) or
        sign extraction (Lion), so effective scale differs from SGD.
    momentum : float
        Momentum coefficient for both Muon and Lion branches.
    ns_steps : int
        Newton-Schulz iterations for Muon (2D+ params).
        Set to 0 for SGD momentum (ablation).
    weight_decay : float
        L2 regularization applied before momentum update. 0 = disabled.

    Returns
    -------
    GradientTransformation
        A (init, update) pair. Compatible with optax's protocol — if you
        later switch to optax, replace GradientTransformation with
        optax.GradientTransformation and nothing else changes.

    Usage
    -----
        opt = mulion(lr=1e-3)
        opt_state = opt.init(params)

        # training step
        grads = jax.grad(loss)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)

    Notes
    -----
    Gradient routing is based on parameter ndim at init time via
    the outer update function — no runtime branching per step.
    """

    def init_fn(params):
        momentum_buf = jax.tree.map(
            lambda p: jnp.zeros_like(p, dtype=jnp.float32),
            params
        )
        return MulionState(
            momentum=momentum_buf,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(grads, state, params=None):
        mom_buf = state.momentum

        def update_one(g, m, p):
            # Cast to fp32 for optimizer arithmetic
            g = g.astype(jnp.float32)

            # Optional weight decay
            if weight_decay > 0.0 and p is not None:
                g = g + weight_decay * p.astype(jnp.float32)

            # Update momentum buffer
            m = momentum * m + (1.0 - momentum) * g

            # Route by dimensionality
            if g.ndim >= 2:
                # Muon branch — orthogonalize momentum matrix
                update = newton_schulz(m, steps=ns_steps)
            else:
                # Lion branch — sign of momentum only
                update = jnp.sign(m)

            return -lr * update, m

        # Apply update_one across pytree
        if params is not None:
            leaves_g, treedef = jax.tree.flatten(grads)
            leaves_m, _ = jax.tree.flatten(mom_buf)
            leaves_p, _ = jax.tree.flatten(params)
            results = [update_one(g, m, p)
                       for g, m, p in zip(leaves_g, leaves_m, leaves_p)]
        else:
            leaves_g, treedef = jax.tree.flatten(grads)
            leaves_m, _ = jax.tree.flatten(mom_buf)
            results = [update_one(g, m, None)
                       for g, m in zip(leaves_g, leaves_m)]

        updates_leaves, new_m_leaves = zip(*results)
        updates = treedef.unflatten(updates_leaves)
        new_mom = treedef.unflatten(new_m_leaves)

        new_state = MulionState(
            momentum=new_mom,
            count=state.count + 1,
        )
        return updates, new_state

    return GradientTransformation(init_fn, update_fn)
