"""
clynx.training
~~~~~~~~~~~~~~
Training utilities.

    TrainState      Immutable pytree: (step, apply_fn, params, opt, opt_state)

Usage
-----
    from clynx.training import TrainState
    from clynx.optim import mulion

    opt   = mulion(lr=1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, opt=opt)

    @jax.jit
    def train_step(state, batch):
        grads = jax.grad(loss_fn)(state.params, batch)
        return state.apply_gradients(grads)
"""

from clynx.training.train_state import TrainState

__all__ = ["TrainState"]
