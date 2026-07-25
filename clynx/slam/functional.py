"""
clynx.slam.functional
~~~~~~~~~~~~~~~~~~~~~
Core primitive operations for SLAM (Super Link Attention Mechanism).

Fixed standard established through experimentation (see project notes):
    - Attention activation : sigmoid only. LinkAttention(Q,K,V) =
      sigmoid(QKᵀ/τ)·V. Softmax's competitive, zero-sum normalization
      needs a different score-scale target than sigmoid's independent
      per-pair evaluation, so this module does not expose a switch —
      pick softmax attention elsewhere if you need it.
    - Gradient stabilization : symlog_bwd hook on every Linear/Embed/
      LinkAttention output. Identity forward, symlog-compressed backward,
      so gradients can't explode across a deep stack regardless of how
      large the raw pre-activation scores get.
    - Optimizer : clynx.optim.mulion (Muon-orthogonalized 2D updates +
      Lion-style sign updates for 1D params) — magnitude-invariant, so
      it doesn't fight the compressed gradients symlog_bwd produces.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def symlog(x: Array) -> Array:
    """
    Symmetric log compression.

        symlog(x) = sign(x) * log(1 + |x|)

    Properties:
    - Monotonic and bijective (information lossless)
    - Sign preserving
    - Linear near zero: symlog(x) ≈ x for |x| << 1
    - Log compression for large |x|
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


# ---------------------------------------------------------------------------
# symlog_bwd — pure gradient-compression hook (identity forward)
# ---------------------------------------------------------------------------
#
# Every module (Linear, Embed, LinkAttention's q/k/v/o projections,
# FeedForward's two Linears, ...) must NOT have its forward values touched
# — a Linear layer needs to stay linear, an Embed lookup needs to stay a
# lookup. What we want is the *backward* half of the symlog trick: compress
# the cotangent flowing through every op via symlog so gradients can't blow
# up as they compound across stacked layers, without ever discarding sign
# or relative order.
#
# symlog_bwd(x) is therefore: identity in the forward pass, symlog(g) in
# the backward pass. Wired into every module's output in this package.

@jax.custom_vjp
def symlog_bwd(x: Array) -> Array:
    """
    Identity forward, SymLog-compressed backward.

    Forward : x                 — unchanged
    Backward: symlog(g)         — cotangent passed through SymLog

    Rationale
    ---------
    Standard autodiff can let gradients compound multiplicatively across a
    deep stack. Passing the incoming cotangent g through symlog instead:
    - Compresses large gradients   → prevents explosion
    - Preserves small gradients    → linear region, no vanishing introduced
    - Preserves sign               → update direction always correct
    - Preserves relative order     → information lossless

    This is a preconditioned gradient, not a true derivative. Verified
    empirically (see project notes): forward pass is bit-identical with the
    hook on/off; backward gradients are consistently (not just noisily)
    smaller in magnitude with the hook on, with compression strength
    growing as |g| grows — negligible for |g|≲1, material once |g|≳3,
    matching symlog's own shape.

    Optimizer compatibility
    -----------------------
    - Mulion (Muon+Lion) OK  orthogonalizes/sign-updates; magnitude irrelevant
    - SGD momentum       ~   affected by magnitude but predictable
    - Adam-family        ~   works (own per-parameter normalization
                              provides a separate, overlapping safety net),
                              but not the combination this package targets
    """
    return x


def _symlog_bwd_hook_fwd(x: Array) -> Tuple[Array, None]:
    return symlog_bwd(x), None


def _symlog_bwd_hook_bwd(_res, g: Array) -> Tuple[Array]:
    return (symlog(g),)


symlog_bwd.defvjp(_symlog_bwd_hook_fwd, _symlog_bwd_hook_bwd)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def linear(x: Array, W: Array, b: Array | None = None) -> Array:
    """
    Linear projection: y = x @ W + b

    Parameters
    ----------
    x : Array, shape (..., in_features)
    W : Array, shape (in_features, out_features)
    b : Array or None, shape (out_features,)
    """
    out = x @ W
    if b is not None:
        out = out + b
    return out


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def embed(E: Array, idx: Array) -> Array:
    """
    Token embedding lookup.

    Parameters
    ----------
    E   : Array, shape (vocab_size, embed_dim)
    idx : Array, integer indices, shape (batch, seq)
    """
    return E[idx]


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.float32) -> Array:
    """
    Causal (autoregressive) mask for self-attention.

    Returns an additive mask of shape (1, 1, seq_len, seq_len) where
    future positions are set to -1e9 (effectively -inf before sigmoid)
    and past/current positions are 0.

    Parameters
    ----------
    seq_len : int
    dtype   : jnp.dtype, default float32

    Returns
    -------
    Array, shape (1, 1, seq_len, seq_len)

    Example
    -------
    mask = F.make_causal_mask(T)
    y    = model.apply(params, x, mask=mask)
    """
    idx = jnp.arange(seq_len)
    # mask[i, j] = 0 if j <= i else -1e9
    mask = jnp.where(idx[None, :] <= idx[:, None], 0.0, -1e9).astype(dtype)
    return mask[None, None, :, :]   # (1, 1, T, T)


def make_attention_mask(
    query_padding: Array,
    key_padding: Array,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """
    Padding mask for cross- or self-attention.

    Parameters
    ----------
    query_padding : Array, shape (batch, q_len)
        1 for real tokens, 0 for padding.
    key_padding   : Array, shape (batch, k_len)
        1 for real tokens, 0 for padding.
    dtype         : jnp.dtype, default float32

    Returns
    -------
    Array, shape (batch, 1, q_len, k_len)
        Additive mask: 0 for valid positions, -1e9 for padding.

    Example
    -------
    pad_mask = F.make_attention_mask(token_ids != 0, token_ids != 0)
    y        = model.apply(params, x, mask=pad_mask)
    """
    # (batch, q_len, k_len): valid where both query and key are real tokens
    mask = jnp.einsum("bi,bj->bij", query_padding, key_padding)
    mask = jnp.where(mask, 0.0, -1e9).astype(dtype)
    return mask[:, None, :, :]   # (B, 1, q_len, k_len)


# ---------------------------------------------------------------------------
# LinkAttention
# ---------------------------------------------------------------------------

def link_attention(
    Q: Array,
    K: Array,
    V: Array,
    tau: Array,
    num_heads: int,
    mask: Array | None = None,
) -> Array:
    """
    SLAM Link Attention (pure function) — sigmoid only.

        LinkAttention(Q, K, V) = sigmoid(QKᵀ / τ) · V

    Each (query, key) pair is scored independently (no softmax-style
    competitive normalization across the key axis), so the temperature
    target is different from scaled dot-product attention: since
    Var(QKᵀ) ∝ d_k, τ should scale ∝ d_k (not ∝ √d_k) to keep each score
    in sigmoid's linear zone regardless of d_k. See LinkAttention's τ
    init in modules.py.

    Parameters
    ----------
    Q, K, V    : Array, shape (batch, seq, d_model)
    tau        : Array, shape (num_heads,) — temperature per head
    num_heads  : int
    mask       : Array or None, shape (batch, 1, seq, seq)
                 Additive mask, large negative where attention suppressed.
                 Use make_causal_mask() or make_attention_mask() to build.

    Returns
    -------
    Array, shape (batch, seq, d_model)
    """
    B, T, d_model = Q.shape
    d_k = d_model // num_heads

    def split_heads(x):
        x = x.reshape(B, T, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    Q = split_heads(Q)  # (B, H, T, d_k)
    K = split_heads(K)
    V = split_heads(V)

    tau = tau[None, :, None, None]

    scores = (Q @ K.transpose(0, 1, 3, 2)) / tau  # (B, H, T, T)

    if mask is not None:
        scores = scores + mask

    attn = jax.nn.sigmoid(scores)
    out = attn @ V                 # (B, H, T, d_k)

    out = out.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    return out


# ---------------------------------------------------------------------------
# Dropout (functional)
# ---------------------------------------------------------------------------

def dropout(x: Array, rate: float, key: Array, deterministic: bool = False) -> Array:
    """
    Inverted dropout.

    Parameters
    ----------
    x             : Array — input
    rate          : float — drop probability in [0, 1)
    key           : Array — JAX PRNGKey (ignored when deterministic=True)
    deterministic : bool  — if True, returns x unchanged (eval mode)

    Returns
    -------
    Array, same shape as x
    """
    if deterministic or rate == 0.0:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, keep, shape=x.shape)
    return jnp.where(mask, x / keep, 0.0)
