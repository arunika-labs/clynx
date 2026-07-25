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
    return_weights: bool = False,
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

    Q may have a different sequence length than K/V (e.g. decoding a
    single new token against a full KV-cache) — only the shared d_model
    and per-head d_k need to match.

    Parameters
    ----------
    Q          : Array, shape (batch, q_len, d_model)
    K, V       : Array, shape (batch, kv_len, d_model)
    tau        : Array, shape (num_heads,) — temperature per head
    num_heads  : int
    mask       : Array or None, shape broadcastable to (batch, num_heads, q_len, kv_len)
                 Additive mask, large negative where attention suppressed.
                 Use make_causal_mask(), make_attention_mask(), or
                 make_cache_mask() to build.
    return_weights : bool, default False
                 If True, also return the attention weight tensor (e.g.
                 for sowing via Module.sow — see modules.LinkAttention).

    Returns
    -------
    Array, shape (batch, q_len, d_model)
        if return_weights=False
    (Array, Array), shapes (batch, q_len, d_model) and (batch, num_heads, q_len, kv_len)
        if return_weights=True — (output, attn_weights)
    """
    B, Tq, d_model = Q.shape
    _, Tk, _ = K.shape
    d_k = d_model // num_heads

    def split_heads(x, T):
        x = x.reshape(B, T, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    Qh = split_heads(Q, Tq)  # (B, H, Tq, d_k)
    Kh = split_heads(K, Tk)  # (B, H, Tk, d_k)
    Vh = split_heads(V, Tk)

    tau = tau[None, :, None, None]

    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / tau  # (B, H, Tq, Tk)

    if mask is not None:
        scores = scores + mask

    attn = jax.nn.sigmoid(scores)
    out = attn @ Vh                # (B, H, Tq, d_k)

    out = out.transpose(0, 2, 1, 3).reshape(B, Tq, d_model)
    if return_weights:
        return out, attn
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


# ---------------------------------------------------------------------------
# RoPE — Rotary Position Embedding
# ---------------------------------------------------------------------------
#
# LinkAttention has no implicit position signal the way softmax attention
# sometimes accidentally picks one up — sigmoid scores each (query, key)
# pair independently, so position has to be injected explicitly. RoPE does
# this by rotating each head's Q/K vectors by an angle proportional to
# absolute position, before the QKᵀ dot product — no extra params, and it
# composes cleanly with a KV-cache since old cached K vectors keep their
# rotation baked in (see LinkAttention(use_rope=True) in modules.py).

def rope_freqs(seq_len: int, d_k: int, base: float = 10000.0, offset: int = 0,
                dtype: jnp.dtype = jnp.float32) -> Tuple[Array, Array]:
    """
    Precompute cos/sin rotation tables for RoPE.

    Parameters
    ----------
    seq_len : int — number of positions to compute (the current chunk length)
    d_k     : int — per-head dimension. Must be even.
    base    : float, default 10000.0 — RoPE frequency base (theta)
    offset  : int, default 0 — absolute position of the first token in this
              chunk. Pass the running cache length when decoding with a
              KV-cache, so newly rotated tokens line up with already-cached
              (already-rotated) ones.
    dtype   : jnp.dtype, default float32

    Returns
    -------
    (cos, sin), each Array of shape (seq_len, d_k)
    """
    assert d_k % 2 == 0, "RoPE requires an even per-head dimension d_k"
    inv_freq = 1.0 / (base ** (jnp.arange(0, d_k, 2, dtype=jnp.float32) / d_k))  # (d_k/2,)
    t = jnp.arange(offset, offset + seq_len, dtype=jnp.float32)                  # (seq_len,)
    freqs = jnp.einsum("i,j->ij", t, inv_freq)                                   # (seq_len, d_k/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)                               # (seq_len, d_k)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def rotate_half(x: Array) -> Array:
    """Rotate-half helper for RoPE: [x1, x2] -> [-x2, x1] (split on last axis)."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """
    Apply RoPE rotation to Q or K.

    Parameters
    ----------
    x   : Array, shape (..., seq_len, d_k) — e.g. (batch, heads, seq_len, d_k)
    cos, sin : Array, shape (seq_len, d_k) — from rope_freqs(); broadcasts
               over any leading batch/head dims of x.

    Returns
    -------
    Array, same shape as x
    """
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# KV-cache — for autoregressive generation
# ---------------------------------------------------------------------------
#
# Fixed-size buffer + a running write-index, kept at a static max_len shape
# so it stays jax.jit-friendly (no dynamically-shaped arrays). New tokens
# are written via jax.lax.dynamic_update_slice_in_dim at the current index;
# unfilled slots are simply masked out of attention via make_cache_mask,
# rather than sliced away, so the buffer shape never changes.
#
# A cache is plain pytree — {"k": ..., "v": ..., "index": ...} — passed
# and returned explicitly (cache in, new_cache out), the same "no hidden
# state" functional style as .init()/.apply() elsewhere in this package.
# See LinkAttention.__call__(cache=...) / TransformerBlock / Stack in
# modules.py for how it's threaded through a model.

def init_kv_cache(batch_size: int, max_len: int, num_heads: int, d_k: int,
                   dtype: jnp.dtype = jnp.float32) -> dict:
    """
    Allocate an empty KV-cache for one LinkAttention layer.

    Parameters
    ----------
    batch_size : int
    max_len    : int — total buffer length (prompt + max new tokens)
    num_heads  : int
    d_k        : int — per-head dimension (d_model // num_heads)
    dtype      : jnp.dtype, default float32

    Returns
    -------
    dict with keys "k" (B,H,max_len,d_k), "v" (B,H,max_len,d_k),
    "index" (scalar int32 — number of positions filled so far)
    """
    shape = (batch_size, num_heads, max_len, d_k)
    return {
        "k": jnp.zeros(shape, dtype=dtype),
        "v": jnp.zeros(shape, dtype=dtype),
        "index": jnp.array(0, dtype=jnp.int32),
    }


def update_kv_cache(cache: dict, k_new: Array, v_new: Array) -> dict:
    """
    Write new K/V for the latest chunk into the cache buffer at cache["index"],
    and advance the index. Does not slice/read — see make_cache_mask() for
    restricting attention to only the filled portion.

    Parameters
    ----------
    cache       : dict — as returned by init_kv_cache() or a previous update_kv_cache()
    k_new, v_new : Array, shape (batch, num_heads, chunk_len, d_k)

    Returns
    -------
    dict — new cache (index advanced by chunk_len)
    """
    chunk_len = k_new.shape[2]
    idx = cache["index"]
    k = jax.lax.dynamic_update_slice_in_dim(cache["k"], k_new.astype(cache["k"].dtype), idx, axis=2)
    v = jax.lax.dynamic_update_slice_in_dim(cache["v"], v_new.astype(cache["v"].dtype), idx, axis=2)
    return {"k": k, "v": v, "index": idx + chunk_len}


def make_cache_mask(offset: Array, chunk_len: int, max_len: int,
                     dtype: jnp.dtype = jnp.float32) -> Array:
    """
    Causal mask for attending from a new chunk of queries against a
    fixed-size KV-cache buffer: query at absolute position (offset + i) may
    attend to any cache slot at absolute position <= (offset + i) — this
    naturally covers both already-cached tokens and the chunk's own tokens
    (self-attention within the chunk stays causal too).

    Parameters
    ----------
    offset    : int or scalar Array — absolute position of the chunk's first token
                (i.e. cache["index"] *before* this chunk was written)
    chunk_len : int — number of new query tokens (1 during single-token decoding)
    max_len   : int — cache buffer length (must match the cache's max_len)
    dtype     : jnp.dtype, default float32

    Returns
    -------
    Array, shape (1, 1, chunk_len, max_len) — additive mask (0 / -1e9)
    """
    query_pos = offset + jnp.arange(chunk_len)   # (chunk_len,) absolute positions
    key_pos = jnp.arange(max_len)                # (max_len,)   absolute slot positions
    mask = jnp.where(key_pos[None, :] <= query_pos[:, None], 0.0, -1e9).astype(dtype)
    return mask[None, None, :, :]   # (1, 1, chunk_len, max_len)
