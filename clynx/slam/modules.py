"""
clynx.slam.modules
~~~~~~~~~~~~~~~~~~
Transformer-complete module system for SLAM (Super Link Attention Mechanism).

API mirrors flax.linen — stateless modules, params as explicit pytree.

    import clynx.slam as nn

    model = nn.LinkAttention(num_heads=8, d_model=512)
    params = model.init(key, x)
    y      = model.apply(params, x)

This package encodes ONE fixed, established standard — not a menu of
switchable options. It was arrived at empirically (see project notes):
sigmoid attention, τ initialized from d_k (not √d_k) and learnable,
symlog-compressed backward gradients everywhere, trained with the Mulion
optimizer (clynx.optim.mulion). There are no activation/norm/tau_learnable
flags to flip — if you need softmax attention or LayerNorm-only variants,
compose them yourself from clynx.slam.functional or plain JAX.

Modules
-------
    Module          Base class (stateless, flax.linen-style)
    Linear          Dense projection, symlog-compressed backward
    Embed           Token embedding lookup, symlog-compressed backward
    LinkAttention   sigmoid(QKᵀ / τ) · V, τ init=d_k, learnable per head
    FeedForward     Two-layer MLP (Linear → activation → Linear)
    Dropout         Inverted dropout (pass deterministic=True at eval)

Normalization: use plain LayerNorm (not included here — it's a handful
of lines and framework-agnostic; SymLogNorm was tried and dropped, see
project notes).
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Any, Callable, Dict
import clynx.slam.functional as F


Params = Dict[str, Any]  # pytree of params


# ---------------------------------------------------------------------------
# Base Module
# ---------------------------------------------------------------------------

class Module:
    """
    Base class for all clynx.slam modules.

    Stateless — all learnable parameters live outside the module as a
    pytree, passed explicitly to init / apply (mirrors flax.linen.Module).

    Subclasses must implement:
        _init(key, *args, **kwargs) -> Params
        _apply(params, *args, **kwargs) -> Array
    """

    def init(self, key: Array, *args, **kwargs) -> Params:
        """Initialize and return a params pytree. Does not mutate the module."""
        return self._init(key, *args, **kwargs)

    def apply(self, params: Params, *args, **kwargs) -> Array:
        """Run the forward pass given params. Does not mutate the module."""
        return self._apply(params, *args, **kwargs)

    def _init(self, key: Array, *args, **kwargs) -> Params:
        raise NotImplementedError

    def _apply(self, params: Params, *args, **kwargs) -> Array:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear(Module):
    """
    Linear projection: y = x @ W + b

    Backward pass is symlog-compressed (see clynx.slam.functional.symlog_bwd)
    so gradients can't explode across a deep stack of Linears.

    Parameters
    ----------
    in_features  : int
    out_features : int
    use_bias     : bool, default True
    """

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.use_bias     = use_bias

    def _init(self, key: Array, *args, **kwargs) -> Params:
        k_w, k_b = jax.random.split(key)
        scale = jnp.sqrt(2.0 / self.in_features)   # Kaiming
        W = jax.random.normal(k_w, (self.in_features, self.out_features)) * scale
        params = {"W": W}
        if self.use_bias:
            params["b"] = jnp.zeros((self.out_features,))
        return params

    def _apply(self, params: Params, x: Array) -> Array:
        out = F.linear(x, params["W"], params.get("b"))
        return F.symlog_bwd(out)


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

class Embed(Module):
    """
    Token embedding lookup. Backward pass is symlog-compressed.

    Parameters
    ----------
    vocab_size : int
    embed_dim  : int
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    def _init(self, key: Array, *args, **kwargs) -> Params:
        E = jax.random.normal(key, (self.vocab_size, self.embed_dim)) * 0.02
        return {"E": E}

    def _apply(self, params: Params, idx: Array) -> Array:
        out = F.embed(params["E"], idx)
        return F.symlog_bwd(out)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout(Module):
    """
    Inverted dropout.

    Unlike learnable modules, Dropout has no params — init returns {}.
    The RNG key is passed at apply time so it's compatible with jit:

        key, subkey = jax.random.split(key)
        x = dropout.apply({}, x, key=subkey, deterministic=False)

    Parameters
    ----------
    rate : float
        Drop probability in [0, 1). 0 = no-op.

    Notes
    -----
    Pass deterministic=True during eval/inference to disable dropout.
    The key argument is ignored when deterministic=True, so you can
    safely pass a dummy key (e.g. jax.random.key(0)).
    """

    def __init__(self, rate: float = 0.1):
        assert 0.0 <= rate < 1.0, "Dropout rate must be in [0, 1)"
        self.rate = rate

    def _init(self, key: Array, *args, **kwargs) -> Params:
        return {}   # no learnable parameters

    def _apply(
        self,
        params: Params,
        x: Array,
        key: Array | None = None,
        deterministic: bool = False,
    ) -> Array:
        if key is None and not deterministic:
            raise ValueError(
                "Dropout.apply requires a PRNGKey when deterministic=False. "
                "Pass key=jax.random.key(n) or set deterministic=True."
            )
        return F.dropout(x, self.rate, key, deterministic=deterministic)


# ---------------------------------------------------------------------------
# LinkAttention
# ---------------------------------------------------------------------------

class LinkAttention(Module):
    """
    SLAM Link Attention.

        LinkAttention(Q, K, V) = sigmoid(QKᵀ / τ) · V

    τ is initialized to d_k (not √d_k) and always learnable.

    Why τ = d_k, not √d_k: softmax's √d_k comes from keeping the RELATIVE
    variance of scores constant before a competitive, zero-sum
    normalization (softmax) across the key axis. Sigmoid has no such
    competition — each (query, key) pair is scored independently, so what
    matters is keeping each score in sigmoid's linear zone in ABSOLUTE
    terms. Since Var(QKᵀ) ∝ d_k, the temperature needs to scale ∝ d_k too,
    to cancel that growth — not ∝ √d_k. d_k is the correct order-of-
    magnitude starting point (empirically found to land within the stable
    basin across every scale tested); τ is left learnable because the
    precise optimal constant co-adapts with model depth and training
    length rather than being a fixed universal number — see project notes.

    Parameters
    ----------
    num_heads : int
    d_model   : int
        Total model dimension. d_k = d_model // num_heads.
    """

    def __init__(self, num_heads: int, d_model: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model   = d_model
        self.d_k       = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def _init(self, key: Array, *args, **kwargs) -> Params:
        keys = jax.random.split(key, 4)
        return {
            "q_proj": self.q_proj.init(keys[0]),
            "k_proj": self.k_proj.init(keys[1]),
            "v_proj": self.v_proj.init(keys[2]),
            "o_proj": self.o_proj.init(keys[3]),
            "tau":    jnp.full((self.num_heads,), float(self.d_k)),
        }

    def _apply(
        self,
        params: Params,
        x: Array,
        mask: Array | None = None,
    ) -> Array:
        """
        Parameters
        ----------
        params : Params
        x      : Array, shape (batch, seq, d_model)
        mask   : Array or None, shape (batch, 1, seq, seq)
                 Use F.make_causal_mask() or F.make_attention_mask().
        """
        Q = self.q_proj.apply(params["q_proj"], x)
        K = self.k_proj.apply(params["k_proj"], x)
        V = self.v_proj.apply(params["v_proj"], x)

        out = F.link_attention(Q, K, V, params["tau"], num_heads=self.num_heads, mask=mask)
        out = F.symlog_bwd(out)
        return self.o_proj.apply(params["o_proj"], out)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(Module):
    """
    Two-layer MLP: Linear → activation → Linear

    Parameters
    ----------
    d_model    : int
    d_ff       : int — hidden dim, typically 4 * d_model
    activation : Callable, default jax.nn.gelu
    """

    def __init__(self, d_model: int, d_ff: int, activation: Callable = jax.nn.gelu):
        self.d_model    = d_model
        self.d_ff       = d_ff
        self.activation = activation

        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def _init(self, key: Array, *args, **kwargs) -> Params:
        k1, k2 = jax.random.split(key)
        return {
            "fc1": self.fc1.init(k1),
            "fc2": self.fc2.init(k2),
        }

    def _apply(self, params: Params, x: Array) -> Array:
        x = self.fc1.apply(params["fc1"], x)
        x = self.activation(x)
        x = self.fc2.apply(params["fc2"], x)
        return x
