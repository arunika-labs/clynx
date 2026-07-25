"""
clynx.optim
~~~~~~~~~~~
Optimizers for SLAM training.

    mulion              Muon (2D+) + Lion (1D) unified optimizer
    MulionState         Optimizer state (momentum + step count)
    GradientTransformation  Minimal optax-compatible (init, update) pair
    newton_schulz       Newton-Schulz orthogonalization (exposed for research)

No external optimizer library required — pure JAX.

Usage
-----
    from clynx.optim import mulion

    opt       = mulion(lr=1e-3, momentum=0.95, ns_steps=5)
    opt_state = opt.init(params)

    # each step:
    updates, opt_state = opt.update(grads, opt_state, params)
    params = jax.tree.map(lambda p, u: p + u, params, updates)
"""

from clynx.optim.mulion import (
    mulion,
    MulionState,
    GradientTransformation,
    newton_schulz,
)

__all__ = [
    "mulion",
    "MulionState",
    "GradientTransformation",
    "newton_schulz",
]
