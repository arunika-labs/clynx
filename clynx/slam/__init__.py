"""
clynx.slam
~~~~~~~~~~
Super Link Attention Mechanism — public API.

One fixed, established standard (see project notes) — sigmoid attention,
τ init=d_k and learnable, symlog-compressed backward everywhere, trained
with clynx.optim.mulion. No activation/norm/tau_learnable switches.

Modules (flax.linen-style, stateless):
    Module          Base class
    Linear          Dense projection, symlog-compressed backward
    Embed           Token embedding lookup, symlog-compressed backward
    LinkAttention   sigmoid(QKᵀ / τ) · V, τ init=d_k, learnable per head
    FeedForward     Two-layer MLP
    Dropout         Inverted dropout (pass deterministic=True at eval)

Functional primitives (clynx.slam.functional):
    symlog              Symmetric log compression
    symlog_bwd          Identity fwd, SymLog-compressed bwd only
    linear              y = x @ W + b
    embed               Token embedding lookup
    link_attention      Core Link Attention op (sigmoid only)
    make_causal_mask    Causal mask for autoregressive models
    make_attention_mask Padding mask for batches with variable length
    dropout             Functional inverted dropout

Usage
-----
    import clynx.slam as nn
    import clynx.slam.functional as F

    model  = nn.LinkAttention(num_heads=8, d_model=512)
    params = model.init(key, x)
    y      = model.apply(params, x, mask=F.make_causal_mask(seq_len))
"""

from clynx.slam.modules import (
    Module,
    Linear,
    Embed,
    LinkAttention,
    FeedForward,
    Dropout,
)

from clynx.slam import functional

from clynx.slam.functional import (
    symlog,
    symlog_bwd,
    make_causal_mask,
    make_attention_mask,
)

__all__ = [
    # Modules
    "Module",
    "Linear",
    "Embed",
    "LinkAttention",
    "FeedForward",
    "Dropout",
    # Submodule
    "functional",
    # Functional shortcuts
    "symlog",
    "symlog_bwd",
    "make_causal_mask",
    "make_attention_mask",
]
