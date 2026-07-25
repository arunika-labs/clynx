"""
clynx.slam.modules
~~~~~~~~~~~~~~~~~~
Transformer-complete module system for SLAM (Super Link Attention Mechanism).

Written on the @compact scoping system (see clynx.slam.scope) — declare and
invoke submodules inline in __call__, flax.linen style, instead of a manual
_init/_apply split. Modules are plain dataclasses — no __init__ to write:

    import clynx.slam as nn

    class MyModel(nn.Module):
        d: int

        @nn.compact
        def __call__(self, x):
            x = nn.LinkAttention(num_heads=8, d_model=self.d)(x)
            return nn.FeedForward(self.d, 4 * self.d)(x)

    model  = MyModel(d=512)
    params = model.init(key, x)
    y      = model.apply(params, x)

setup() is also available as the other flax.linen module-building style —
assign submodules as attributes, call them from a plain __call__ — see
clynx.slam.scope.Module for the full example.

Add @nn.pack on top for PyTorch/Equinox-style direct calling instead of
the explicit init/apply split — see clynx.slam.scope.pack.

This package encodes ONE fixed, established standard — not a menu of
switchable options (see project notes): sigmoid attention, τ initialized
from d_k (not √d_k) and learnable, symlog-compressed backward gradients
everywhere, trained with the Mulion optimizer (clynx.optim.mulion). There
are no activation/norm/tau_learnable flags — if you need softmax attention
or LayerNorm-only variants, compose them yourself.

Modules
-------
    Module          Base class — see clynx.slam.scope
    Dense           Dense (linear) projection, symlog-compressed backward.
                    `Linear` is kept as an alias (same class) for anyone
                    used to that name — flax.linen calls this Dense.
    Embed           Token embedding lookup, symlog-compressed backward
    LinkAttention   sigmoid(QKᵀ / τ) · V, τ init=d_k, learnable per head.
                    Optional RoPE (use_rope=True) and KV-cache (cache=...)
                    for autoregressive generation.
    FeedForward     Two-layer MLP (Linear → activation → Linear)
    Dropout         Inverted dropout (pass deterministic=True at eval)
    LayerNorm       Standard LayerNorm — TransformerBlock's default norm,
                    also fine standalone (e.g. a final pre-head norm)
    TransformerBlock  LinkAttention + FeedForward, pre-norm + residuals
    Sequential      Chain a fixed list of modules, flax.linen-style
    Stack           Repeat a block factory N times (e.g. N TransformerBlocks),
                    threading per-layer KV-caches automatically

Embed also has .attend(query) for a weight-tied LM head (query @ E.T),
mirroring flax.linen.Embed.attend.

Normalization: LayerNorm above is the one norm primitive this package
ships (a handful of lines, framework-agnostic — see project notes);
TransformerBlock takes norm_cls=... to swap in RMSNorm or anything else.
"""

import dataclasses
import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, Sequence as SequenceType

import clynx.slam.functional as F
from clynx.slam.scope import Module, compact, pack, current_scope


def _has_dropout_rng() -> bool:
    """True if a named 'dropout' RNG stream is available in the current scope."""
    scope = current_scope()
    return scope is not None and scope.rngs is not None and "dropout" in scope.rngs


__all__ = ["Module", "compact", "pack", "Dense", "Linear", "Embed", "LinkAttention",
           "FeedForward", "Dropout", "LayerNorm", "TransformerBlock", "Sequential", "Stack"]


# ---------------------------------------------------------------------------
# Dense (a.k.a. Linear)
# ---------------------------------------------------------------------------

class Dense(Module):
    """
    Dense (linear) projection: y = x @ W + b — same as flax.linen.Dense.

    Backward pass is symlog-compressed (see clynx.slam.functional.symlog_bwd)
    so gradients can't explode across a deep stack.

    Fields
    ------
    in_features  : int
    out_features : int
    use_bias     : bool, default True
    name         : str or None — explicit param-tree key when used as a
                   submodule; auto-numbered ("Dense_0", "Dense_1", ...)
                   by call order if omitted.
    """

    in_features: int
    out_features: int
    use_bias: bool = True

    @compact
    def __call__(self, x: Array) -> Array:
        scale = jnp.sqrt(2.0 / self.in_features)   # Kaiming
        W = self.param("W", lambda k: jax.random.normal(
            k, (self.in_features, self.out_features)) * scale)
        b = self.param("b", lambda k: jnp.zeros((self.out_features,))) if self.use_bias else None
        out = F.linear(x, W, b)
        return F.symlog_bwd(out)


# Alias for anyone used to the old name / PyTorch-style naming. Same class,
# not a subclass — Linear IS Dense, so isinstance checks, param trees
# (top-level auto-name would read "Dense_i" either way since __name__ is
# unchanged), and .copy() all behave identically either way you spell it.
Linear = Dense


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

class Embed(Module):
    """
    Token embedding lookup. Backward pass is symlog-compressed.

    Fields
    ------
    vocab_size : int
    embed_dim  : int
    name       : str or None
    """

    vocab_size: int
    embed_dim: int

    @compact
    def __call__(self, idx: Array) -> Array:
        E = self.param("E", lambda k: jax.random.normal(
            k, (self.vocab_size, self.embed_dim)) * 0.02)
        out = F.embed(E, idx)
        return F.symlog_bwd(out)

    def attend(self, query: Array) -> Array:
        """
        Tied output head: score `query` against every embedding row —
        query @ E.T, shape (..., vocab_size). Call after this Embed has
        already been invoked at least once in the SAME enclosing
        @compact __call__ (so its params exist in the current scope) —
        same idea as flax.linen.Embed.attend, for a language-model head
        that shares weights with the input embedding instead of a
        separate Linear(d_model, vocab_size).

        Requires an explicit name=... on this Embed (auto-numbered names
        aren't reliable to look up outside the normal call path):

            embed  = nn.Embed(vocab_size, d_model, name="tok_embed")
            x      = embed(idx)          # creates params under "tok_embed"
            ...
            logits = embed.attend(x)     # reads the same E, transposed
        """
        if self._name is None:
            raise RuntimeError(
                "Embed.attend() requires this Embed to have an explicit "
                "name=..., e.g. nn.Embed(vocab, dim, name='tok_embed'), so "
                "its params can be found reliably."
            )
        parent = current_scope()
        if parent is None:
            raise RuntimeError(
                "Embed.attend() called outside a scope — call it from "
                "inside a @compact __call__, after this Embed's __call__ "
                "has already run once in the same scope."
            )
        if self._name not in parent.params or "E" not in parent.params[self._name]:
            raise KeyError(
                f"No params found for Embed '{self._name}' yet — call "
                f"this Embed once (e.g. embed(idx)) before .attend()."
            )
        E = parent.params[self._name]["E"]
        return query @ E.T


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout(Module):
    """
    Inverted dropout. Has no learnable params — init() on a bare Dropout
    returns {}.

    Fields
    ------
    rate : float, default 0.1
        Drop probability in [0, 1). 0 = no-op.
    name : str or None

    Notes
    -----
    Pass deterministic=True during eval/inference to disable dropout. The
    key argument is call-time (not a stored param) — pass a fresh PRNGKey
    split every step, same as clynx.slam.functional.dropout. Prefer
    self.make_rng("dropout") (see clynx.slam.scope.Module.make_rng) over
    threading a key manually if this Dropout is used as a submodule inside
    a larger model's rngs=... call.
    """

    rate: float = 0.1

    def __post_init__(self):
        super().__post_init__()   # sets self._name, per Module
        assert 0.0 <= self.rate < 1.0, "Dropout rate must be in [0, 1)"

    @compact
    def __call__(self, x: Array, key: Array | None = None, deterministic: bool = False) -> Array:
        if key is None and not deterministic:
            key = self.make_rng("dropout") if _has_dropout_rng() else None
        if key is None and not deterministic:
            raise ValueError(
                "Dropout requires a PRNGKey when deterministic=False. "
                "Pass key=jax.random.key(n) explicitly, or supply "
                "rngs={'dropout': key} to .init()/.apply() so "
                "self.make_rng('dropout') can provide one automatically."
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

    Fields
    ------
    num_heads : int
    d_model   : int
        Total model dimension. d_k = d_model // num_heads (computed
        automatically — don't pass it, it's not a constructor argument).
    use_rope  : bool, default False
        Rotate Q/K per-head before scoring (see clynx.slam.functional.apply_rope).
        LinkAttention has no other source of position information, so this
        is normally what you want for anything beyond a single token.
    rope_base : float, default 10000.0
        RoPE frequency base — only used if use_rope=True.
    name      : str or None
    """

    num_heads: int
    d_model: int
    use_rope: bool = False
    rope_base: float = 10000.0
    d_k: int = dataclasses.field(init=False)

    def __post_init__(self):
        super().__post_init__()   # sets self._name, per Module
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // self.num_heads

    @compact
    def __call__(self, x: Array, mask: Array | None = None, cache: dict | None = None,
                 offset: int = 0):
        """
        Parameters
        ----------
        x      : Array, shape (batch, seq, d_model)
        mask   : Array or None. Without cache: shape (batch, 1, seq, seq),
                 use F.make_causal_mask() or F.make_attention_mask(). With
                 cache: shape (..., seq, cache_max_len), use
                 F.make_cache_mask() — or leave None and one is built for
                 you from `offset`/`cache` (plain causal-into-cache).
        cache  : dict or None. Pass a KV-cache from F.init_kv_cache() (or a
                 previous call's returned cache) to attend against past
                 tokens without recomputing them. `x` should then be just
                 the NEW token(s), not the full sequence. When cache is
                 given, this returns (output, new_cache) instead of just
                 output.
        offset : int, default 0. Absolute position of x's first token —
                 required for correct RoPE / cache-masking when x is a
                 continuation rather than the start of a sequence. If
                 `cache` is given and offset is left at 0, cache["index"]
                 (the cache's own running position) is used automatically.

        Returns
        -------
        Array, shape (batch, seq, d_model)          if cache is None
        (Array, dict)                                if cache is not None
        """
        Q = Dense(self.d_model, self.d_model, name="q_proj")(x)
        K = Dense(self.d_model, self.d_model, name="k_proj")(x)
        V = Dense(self.d_model, self.d_model, name="v_proj")(x)

        tau = self.param("tau", lambda k: jnp.full((self.num_heads,), float(self.d_k)))

        pos_offset = cache["index"] if (cache is not None and offset == 0) else offset

        if self.use_rope:
            B, T, _ = Q.shape

            def split(a):
                return a.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

            def merge(a):
                return a.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

            cos, sin = F.rope_freqs(T, self.d_k, base=self.rope_base, offset=pos_offset)
            Q = merge(F.apply_rope(split(Q), cos, sin))
            K = merge(F.apply_rope(split(K), cos, sin))

        new_cache = None
        if cache is not None:
            B, T, _ = K.shape

            def split(a):
                return a.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

            new_cache = F.update_kv_cache(cache, split(K), split(V))
            max_len = new_cache["k"].shape[2]

            def merge(a):
                return a.transpose(0, 2, 1, 3).reshape(B, max_len, self.d_model)

            K = merge(new_cache["k"])
            V = merge(new_cache["v"])
            if mask is None:
                mask = F.make_cache_mask(pos_offset, T, max_len)

        out, attn_weights = F.link_attention(
            Q, K, V, tau, num_heads=self.num_heads, mask=mask, return_weights=True)
        # No-op unless the caller passed mutable=["intermediates"] to
        # .apply()/.init_with_output() — see Module.sow() in scope.py.
        self.sow("intermediates", "attn_weights", attn_weights)
        out = F.symlog_bwd(out)
        out = Dense(self.d_model, self.d_model, name="o_proj")(out)

        if cache is not None:
            return out, new_cache
        return out


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(Module):
    """
    Two-layer MLP: Dense → activation → Dense

    Fields
    ------
    d_model    : int
    d_ff       : int — hidden dim, typically 4 * d_model
    activation : Callable, default jax.nn.gelu
    name       : str or None
    """

    d_model: int
    d_ff: int
    activation: Callable = jax.nn.gelu

    @compact
    def __call__(self, x: Array) -> Array:
        x = Dense(self.d_model, self.d_ff, name="fc1")(x)
        x = self.activation(x)
        x = Dense(self.d_ff, self.d_model, name="fc2")(x)
        return x


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
#
# The one normalization primitive this package DOES ship — "a handful of
# lines, framework-agnostic" (see project notes) — used as TransformerBlock's
# default (pass norm_cls=... to swap it for something else there) and
# equally fine to use standalone, e.g. a final norm before an LM head:
#
#     x = nn.Stack(..., num_layers=12)(x, mask=mask)
#     x = nn.LayerNorm(name="norm_f")(x)
#     logits = nn.Linear(d_model, vocab_size)(x)

class LayerNorm(Module):
    """
    Standard LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta,
    normalized over the last axis.

    Fields
    ------
    eps  : float, default 1e-6
    name : str or None
    """

    eps: float = 1e-6

    @compact
    def __call__(self, x: Array) -> Array:
        dim = x.shape[-1]
        gamma = self.param("gamma", lambda k: jnp.ones((dim,)))
        beta = self.param("beta", lambda k: jnp.zeros((dim,)))
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps) * gamma + beta


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(Module):
    """
    One pre-norm transformer block: x + LinkAttention(norm(x)), then
    x + FeedForward(norm(x)) — the standard residual wiring, built from
    the pieces above.

        block  = nn.TransformerBlock(num_heads=8, d_model=512, d_ff=2048)
        params = block.init(key, x)
        y      = block.apply(params, x, mask=F.make_causal_mask(T))

    Fields
    ------
    num_heads     : int
    d_model       : int
    d_ff          : int — FeedForward hidden dim, typically 4 * d_model
    use_rope      : bool, default False — see LinkAttention
    rope_base     : float, default 10000.0
    dropout_rate  : float, default 0.0 — applied after attention and after
                    the feedforward, each with its own Dropout instance
    activation    : Callable, default jax.nn.gelu — FeedForward's activation
    norm_cls      : Module subclass or None, default None — normalization
                    module used before attention/feedforward (called with
                    no constructor args beyond `name`). Defaults to a
                    minimal built-in LayerNorm; pass your own (e.g. an
                    RMSNorm) to override.
    name          : str or None

    Call signature
    --------------
    __call__(x, mask=None, cache=None, offset=0, deterministic=True, dropout_key=None)

    Same cache/offset/mask semantics as LinkAttention — pass cache=... to
    get (output, new_cache) back for autoregressive generation. Pass
    deterministic=False and a dropout_key to enable dropout during training.
    """

    num_heads: int
    d_model: int
    d_ff: int
    use_rope: bool = False
    rope_base: float = 10000.0
    dropout_rate: float = 0.0
    activation: Callable = jax.nn.gelu
    norm_cls: type | None = None

    @compact
    def __call__(self, x: Array, mask: Array | None = None, cache: dict | None = None,
                 offset: int = 0, deterministic: bool = True, dropout_key: Array | None = None):
        Norm = self.norm_cls or LayerNorm
        drop_key1, drop_key2 = (jax.random.split(dropout_key) if dropout_key is not None
                                 else (None, None))

        attn_in = Norm(name="norm1")(x)
        attn_out = LinkAttention(
            num_heads=self.num_heads, d_model=self.d_model,
            use_rope=self.use_rope, rope_base=self.rope_base, name="attn",
        )(attn_in, mask=mask, cache=cache, offset=offset)

        new_cache = None
        if cache is not None:
            attn_out, new_cache = attn_out

        attn_out = Dropout(self.dropout_rate, name="drop1")(
            attn_out, key=drop_key1, deterministic=deterministic)
        x = x + attn_out

        ff_out = FeedForward(self.d_model, self.d_ff, activation=self.activation,
                              name="ff")(Norm(name="norm2")(x))
        ff_out = Dropout(self.dropout_rate, name="drop2")(
            ff_out, key=drop_key2, deterministic=deterministic)
        x = x + ff_out

        if cache is not None:
            return x, new_cache
        return x


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------

class Sequential(Module):
    """
    Chain a fixed list of modules, each fed the previous one's output —
    flax.linen-style nn.Sequential.

        model = nn.Sequential([
            nn.Linear(512, 2048, name="up"),
            nn.Linear(2048, 512, name="down"),
        ])

    Only the first module receives extra *args/**kwargs on the call; every
    later module is called with just its predecessor's output. For anything
    needing per-layer arguments (masks, caches, ...), use Stack instead.

    Fields
    ------
    modules : Sequence[Module] — must have distinct `name`s if types repeat
              (auto-numbering by class name still applies otherwise).
    name    : str or None
    """

    modules: SequenceType[Module] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()   # sets self._name, per Module
        self.modules = list(self.modules)

    @compact
    def __call__(self, x, *args, **kwargs):
        for i, m in enumerate(self.modules):
            x = m(x, *args, **kwargs) if i == 0 else m(x)
        return x


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

class Stack(Module):
    """
    Repeat a block factory N times, e.g. N TransformerBlocks — and thread a
    per-layer list of KV-caches through automatically if you're generating.

        stack  = nn.Stack(lambda: nn.TransformerBlock(num_heads=8, d_model=512, d_ff=2048),
                           num_layers=12)
        params = stack.init(key, x)
        y      = stack.apply(params, x, mask=F.make_causal_mask(T))

        # generation with cache: one dict per layer
        caches = [F.init_kv_cache(B, max_len, num_heads, d_model // num_heads)
                  for _ in range(12)]
        y, caches = stack.apply(params, x_new, cache=caches, offset=t)

    Fields
    ------
    block_fn   : Callable[[], Module] — factory called once per layer (fresh
                 instance each time — needed for correct auto-naming). Must
                 build a module whose __call__(x, mask=None, cache=None,
                 offset=0, ...) matches TransformerBlock's signature if you
                 intend to use masks/caches.
    num_layers : int
    name       : str or None
    """

    block_fn: Callable[[], Module]
    num_layers: int

    @compact
    def __call__(self, x, mask: Array | None = None, cache=None, offset: int = 0, **kwargs):
        new_caches = [] if cache is not None else None
        for i in range(self.num_layers):
            layer_cache = cache[i] if cache is not None else None
            out = self.block_fn()(x, mask=mask, cache=layer_cache, offset=offset, **kwargs)
            if cache is not None:
                x, nc = out
                new_caches.append(nc)
            else:
                x = out
        if cache is not None:
            return x, new_caches
        return x
