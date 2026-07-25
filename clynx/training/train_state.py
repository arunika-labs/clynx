"""
clynx.training.train_state
~~~~~~~~~~~~~~~~~~~~~~~~~~
TrainState — a single pytree that bundles params, optimizer state, and
step counter so they can never go out of sync.

    state = TrainState.create(params=params, opt=opt)

    # training step (jit-safe):
    grads = jax.grad(loss_fn)(state.params, batch)
    state = state.apply_gradients(grads)

Design notes
------------
- Pure NamedTuple so the whole state is a valid JAX pytree — can be
  passed to jax.jit, jax.pmap, jax.lax.scan without wrapping.
- apply_fn is stored as a non-pytree field (static) so jit doesn't try
  to trace through the Python function object. Access via state.apply_fn.
- No dependency on optax; works with clynx.optim.mulion out of the box.
  Compatible with any optimizer that follows the (init, update) protocol.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Any, Callable, NamedTuple


# ---------------------------------------------------------------------------
# _StaticFn: wraps apply_fn so JAX treats it as a static (non-array) field
# ---------------------------------------------------------------------------

class _StaticFn:
    """
    Thin wrapper so apply_fn survives jax.tree.map without being traced.
    Registered as a pytree with no leaves, so JAX sees it as a static node.
    """
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"_StaticFn({self.fn})"


jax.tree_util.register_pytree_node(
    _StaticFn,
    lambda x: ([], x.fn),           # no leaves, fn is aux data
    lambda fn, _: _StaticFn(fn),    # reconstruct from aux
)


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

class TrainState(NamedTuple):
    """
    Immutable training state: (step, apply_fn, params, opt, opt_state).

    Fields
    ------
    step      : int scalar — incremented by every apply_gradients call
    apply_fn  : _StaticFn wrapping model.apply — call via state.apply_fn(...)
    params    : pytree of model parameters
    opt       : GradientTransformation (init, update) — stored as static node
    opt_state : pytree of optimizer state (momentum, step counter, ...)

    Usage
    -----
        model  = nn.LinkAttention(num_heads=8, d_model=512)
        params = model.init(key, x)
        opt    = mulion(lr=1e-3)

        state  = TrainState.create(
            apply_fn = model.apply,
            params   = params,
            opt      = opt,
        )

        # inside jit:
        @jax.jit
        def train_step(state, batch):
            def loss_fn(params):
                logits = state.apply_fn(params, batch['x'])
                return cross_entropy(logits, batch['y'])
            grads = jax.grad(loss_fn)(state.params)
            return state.apply_gradients(grads)

        state = train_step(state, batch)
        print(state.step)   # 1
    """

    step      : Array               # shape (), int32
    apply_fn  : _StaticFn          # model.apply, static
    params    : Any                 # pytree
    opt       : Any                 # GradientTransformation, static
    opt_state : Any                 # pytree

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable,
        params: Any,
        opt: Any,
    ) -> "TrainState":
        """
        Create a new TrainState at step 0.

        Parameters
        ----------
        apply_fn : Callable — model.apply (or any params -> output function)
        params   : pytree  — initial model parameters
        opt      : GradientTransformation — e.g. mulion(lr=1e-3)
        """
        opt_state = opt.init(params)
        return cls(
            step      = jnp.zeros([], jnp.int32),
            apply_fn  = _StaticFn(apply_fn),
            params    = params,
            opt       = opt,
            opt_state = opt_state,
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def apply_gradients(self, grads: Any) -> "TrainState":
        """
        Apply gradients and return a new TrainState.

        Parameters
        ----------
        grads : pytree matching self.params — output of jax.grad

        Returns
        -------
        New TrainState with step+1, updated params and opt_state.

        Notes
        -----
        This is a pure function — self is not mutated. Assign the return
        value: ``state = state.apply_gradients(grads)``
        """
        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)
        new_params = jax.tree.map(lambda p, u: p + u, self.params, updates)
        return self._replace(
            step      = self.step + 1,
            params    = new_params,
            opt_state = new_opt_state,
        )
