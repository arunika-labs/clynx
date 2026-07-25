"""
clynx
~~~~~
Super Link Attention Mechanism (SLAM) + Mulion optimizer.

Subpackages
-----------
    clynx.slam        Transformer modules and functional primitives
    clynx.optim       Mulion optimizer (Muon + Lion, no external dep)
    clynx.training    TrainState and training utilities

Quick start
-----------
    import clynx.slam as nn
    import clynx.slam.functional as F
    from clynx.optim import mulion
    from clynx.training import TrainState

    model     = nn.LinkAttention(num_heads=8, d_model=512)
    params    = model.init(key, x)
    opt       = mulion(lr=1e-3)
    state     = TrainState.create(apply_fn=model.apply, params=params, opt=opt)

    @jax.jit
    def train_step(state, batch):
        grads = jax.grad(loss_fn)(state.params, batch)
        return state.apply_gradients(grads)
"""

from clynx import slam, optim, training, serialize

__version__ = "0.1.0"
__all__ = ["slam", "optim", "training", "serialize"]
