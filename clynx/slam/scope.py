"""
clynx.slam.scope
~~~~~~~~~~~~~~~~
The param-tracking machinery behind @compact and @pack.

@compact lets a Module's __call__ declare AND invoke submodules inline
(flax.linen style) instead of hand-writing a separate _init/_apply split:

    class MyModel(nn.Module):
        d: int

        @nn.compact
        def __call__(self, x):
            x = nn.Linear(self.d, self.d, name="in_proj")(x)
            x = nn.LinkAttention(num_heads=8, d_model=self.d)(x)
            return nn.Linear(self.d, self.d, name="out_proj")(x)

    params = model.init(key, x)   # walks the __call__ once, builds the tree
    y      = model.apply(params, x)

@pack is an OPTIONAL class decorator on top of that, for people who want
PyTorch/Equinox-style ergonomics instead of the explicit init/apply split:

    @nn.pack
    class MyModel(nn.Module):
        d: int
        @nn.compact
        def __call__(self, x): ...

    model = MyModel(d=512)
    out   = model(x)          # lazily inits on first call, reuses after

How it works
------------
A single ContextVar holds the "current Scope" while a module tree is being
walked (either building params during .init(), or reading them during
.apply()). Each Module, when invoked as a submodule of another module's
@compact call, gets its own child Scope: a dict slice of the parent's
params, addressed by name (explicit `name=...` or auto-numbered by class
name and call order — deterministic as long as the same code path runs
both at init time and at apply time, exactly like flax.linen).

@pack is intentionally a thin convenience wrapper, not a new execution
mode: it just calls .init() once (lazily, on first invocation) and .apply()
on every call, storing the resulting params as instance state. For actual
training (extracting params, computing grads, writing updated params back)
use .params / .apply() directly — see Module.params docs below. Packed
instances are fine to READ inside jax.jit/jax.grad (their stored params
are ordinary arrays closed over as constants), but the packed __call__
itself is not what you differentiate through for a training step; extract
`.params` and use `.apply(params, x)` in your loss function instead.
"""

from __future__ import annotations
import contextvars
import dataclasses
import functools
import inspect
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional, Tuple


_current_scope: contextvars.ContextVar[Optional["Scope"]] = contextvars.ContextVar(
    "clynx_current_scope", default=None
)


class Scope:
    """One node in the param tree currently being built (init) or read (apply)."""

    __slots__ = ("mode", "params", "rng", "rngs", "counters", "name", "owner",
                 "mutable", "collections", "variables")

    def __init__(self, mode: str, params: Dict[str, Any], rng, name: str, owner: Any,
                 mutable: FrozenSet[str] = frozenset(), rngs: Optional[Dict[str, Any]] = None,
                 variables: Optional[Dict[str, Dict[str, Any]]] = None):
        assert mode in ("init", "apply")
        self.mode = mode
        self.params = params      # init: dict being filled in. apply: dict being read.
        self.rng = rng            # PRNGKey (init only) or None (apply)
        self.rngs = rngs          # optional named RNG streams, e.g. {"dropout": key} — usable in both init and apply
        self.counters: Dict[str, int] = {}
        self.name = name
        self.owner = owner        # the Module instance this scope belongs to

        # Side-channel collections other than "params" (e.g. "intermediates",
        # written via Module.sow()). Only writable for collection names
        # listed in `mutable` — this scope's own slice, merged back into the
        # parent's `collections[col][name]` by the @compact wrapper, mirroring
        # how `params` is merged back during .init().
        self.mutable = mutable
        self.collections: Dict[str, Dict[str, Any]] = {}

        # Mutable *read-write* variable collections (e.g. "batch_stats"),
        # created/read via Module.variable() — unlike sow() (write-only,
        # accumulates for inspection), variable() values can be read back
        # and reassigned within the same or a later apply() call. Keyed by
        # collection name -> this scope's slice of that collection.
        self.variables: Dict[str, Dict[str, Any]] = variables if variables is not None else {}

    def next_rng(self):
        if self.rng is None:
            raise RuntimeError(
                "No RNG available in this scope. This happens if a param is "
                "created during .apply() instead of .init() — params must be "
                "fully created during .init()."
            )
        self.rng, sub = jax.random.split(self.rng)
        return sub

    def next_named_rng(self, name: str):
        """Split off a fresh key from the named RNG stream `name` (e.g. "dropout")."""
        streams = self.rngs
        if streams is None or name not in streams:
            raise RuntimeError(
                f"No RNG stream named '{name}' available in this scope. Pass "
                f"it via rngs={{'{name}': key, ...}} to .init()/.apply()."
            )
        streams[name], sub = jax.random.split(streams[name])
        return sub

    def auto_name(self, cls_name: str) -> str:
        n = self.counters.get(cls_name, 0)
        self.counters[cls_name] = n + 1
        return f"{cls_name}_{n}"


def _push(scope: Scope):
    return _current_scope.set(scope)


def _pop(token):
    _current_scope.reset(token)


def current_scope() -> Optional[Scope]:
    return _current_scope.get()


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------

class Module:
    """
    Base class for clynx.slam modules — dataclass-style config, flax.linen
    style, using the same @compact scoping system as before.

    Declare config as class-level type-annotated fields, no __init__ needed:

        class LinkAttention(nn.Module):
            num_heads: int
            d_model: int
            use_rope: bool = False

            @nn.compact
            def __call__(self, x): ...

    Every subclass is automatically turned into a dataclass (like
    flax.linen.Module) by __init_subclass__ below — you only write
    __init__ yourself for unusual cases (e.g. validation logic beyond a
    plain field default), same escape hatch flax.linen leaves open.

    A `name: Optional[str] = None` field is injected into every subclass
    automatically (skipped if you already declared your own `name` field),
    exactly like flax.linen's implicit `name`. Actual learnable arrays are
    created via self.param(...)/self.variable(...) or by invoking child
    Module instances — plain fields above are config only.

    setup() vs @compact
    --------------------
    Two ways to build submodules, same as flax.linen:

    - @compact __call__ (as above): declare AND invoke submodules inline,
      in one method. A Module may have at most one @compact method.
    - setup(): assign submodules (and self.variable/self.param-backed
      values you want named ahead of time) as attributes, then reference
      them from a plain (non-@compact) __call__:

        class MyModel(nn.Module):
            d: int

            def setup(self):
                self.dense1 = Linear(self.d, self.d, name="in_proj")
                self.attn   = LinkAttention(num_heads=8, d_model=self.d)

            def __call__(self, x):
                x = self.dense1(x)
                return self.attn(x)

    Don't mix both on the same class — define @compact __call__ OR
    setup()+__call__, not both.
    """

    # NOTE: `name` is NOT declared here as a dataclass field on the base
    # class — dataclasses require fields-without-defaults to precede
    # fields-with-defaults, and since Module itself is never decorated with
    # @dataclass (only subclasses are), inheriting a defaulted `name` field
    # from here would break any subclass that declares a required field
    # (e.g. `num_heads: int` with no default). Instead __init_subclass__
    # below injects `name: Optional[str] = None` directly into EACH
    # subclass's own __annotations__, appended after that subclass's own
    # fields, so it always lands last in field order — safe regardless of
    # what other fields (required or defaulted) the subclass declares.
    name: Optional[str] = None  # plain class attribute fallback for introspection only

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Snapshot whatever __call__ is at class-body-execution time (i.e.
        # the @compact-wrapped dispatch, or a plain setup()-style __call__)
        # BEFORE any class decorator like @pack gets a chance to overwrite
        # __call__ for stateful ergonomics, and BEFORE @dataclass below can
        # touch it. .init()/.apply() always go through this snapshot, never
        # through self(...), so @pack can freely redefine __call__ without
        # breaking the underlying functional init/apply machinery or
        # causing infinite recursion.
        raw_call = cls.__dict__.get("__call__")
        has_setup = "setup" in cls.__dict__
        if raw_call is not None:
            is_compact_wrapped = getattr(raw_call, "_is_compact", False)
            if has_setup and not is_compact_wrapped:
                # setup()-style: __call__ is a PLAIN method (not @compact).
                # Wrap so setup() runs once, inside the scope, before the
                # real __call__ body — same scoping semantics as @compact,
                # just without requiring the decorator on __call__ itself.
                cls._compact_dispatch = _make_setup_dispatch(raw_call)
            else:
                cls._compact_dispatch = raw_call
        elif has_setup:
            raise TypeError(
                f"{cls.__name__} defines setup() but no __call__ — add a "
                f"plain (non-@compact) __call__ method that uses the "
                f"submodules/params you assign in setup()."
            )

        # Turn the subclass into a dataclass, flax.linen-style, UNLESS it
        # already defines its own __init__ (escape hatch for unusual cases)
        # or has already been processed (avoid double-decoration if some
        # intermediate base in the MRO already ran this).
        if "__init__" in cls.__dict__:
            return  # user opted out of auto-dataclass by writing their own __init__

        # Only annotate a `name` field if this exact class doesn't already
        # declare one itself.
        annotations = cls.__dict__.get("__annotations__", {})
        if "name" not in annotations:
            annotations["name"] = Optional[str]
            cls.__annotations__ = annotations
            if "name" not in cls.__dict__:
                cls.name = None

        cls = dataclasses.dataclass(eq=False)(cls)
        # Keep this exact subclass object (dataclass() mutates in place and
        # returns the same class), nothing further to reassign.

    def __post_init__(self):
        # Alias so existing internal code (and any subclass) can keep using
        # self._name — the public field is `name`, matching flax.linen.
        self._name = self.name

    def setup(self) -> None:
        """
        Override to assign submodules (and other config-derived state) as
        attributes, for use by a plain (non-@compact) __call__ — the other
        flax.linen module-building style, alongside @compact. No-op by
        default. See the Module docstring above for a full example.
        """
        pass

    # -- functional entry points ------------------------------------------------

    def init(self, key, *args, mutable: Iterable[str] = (),
              rngs: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Build and return a fresh params pytree by walking __call__ once.

        `mutable` rarely matters at init time (sow()'d collections are
        normally only inspected during .apply()), but is accepted for
        symmetry — see init_with_output() if you also need the output.

        `rngs` optionally supplies named RNG streams (e.g. {"dropout": key})
        for use via self.make_rng("dropout") inside __call__, separate from
        the `key` used to initialize params.

        If any self.variable(...) calls happen during this init pass, their
        initial values are collected too — call init_with_output() (or
        .init(..., mutable=[...]) is NOT enough on its own; use
        init_with_output for the extra return value) if you need them back
        immediately. Otherwise they're simply available on the returned
        params-adjacent state the first time you .apply(..., mutable=[...]).
        """
        dispatch = self._require_dispatch()
        root = Scope("init", {}, key, name="", owner=self, mutable=frozenset(mutable), rngs=rngs)
        token = _push(root)
        try:
            dispatch(self, *args, **kwargs)
        finally:
            _pop(token)
        if root.mutable and root.variables:
            return {"params": root.params, **root.variables}
        return root.params

    def init_with_output(self, key, *args, mutable: Iterable[str] = (),
                          rngs: Optional[Dict[str, Any]] = None, **kwargs):
        """Like .init(), but also returns the __call__ output from that same pass.

        Saves a redundant .apply() when you just want to inspect the output
        of the params you're about to create (e.g. checking shapes).

            y, params = model.init_with_output(key, x)

        Returns
        -------
        (output, params)                          if mutable is empty
        (output, params, state)                    if mutable is non-empty,
            where `state` merges any sow()'d collections (e.g.
            "intermediates") and any self.variable(...) collections (e.g.
            "batch_stats") created during this pass.
        """
        dispatch = self._require_dispatch()
        root = Scope("init", {}, key, name="", owner=self, mutable=frozenset(mutable), rngs=rngs)
        token = _push(root)
        try:
            out = dispatch(self, *args, **kwargs)
        finally:
            _pop(token)
        if root.mutable:
            state = {**root.collections, **root.variables}
            return out, root.params, state
        return out, root.params

    def apply(self, params: Dict[str, Any], *args, mutable: Iterable[str] = (),
              rngs: Optional[Dict[str, Any]] = None, **kwargs):
        """Run the forward pass using an existing params pytree.

        Pass mutable=("intermediates",) (or any collection name(s) sown via
        self.sow(...), or read/written via self.variable(...), inside
        __call__) to also get those values back:

            y                    = model.apply(params, x)
            y, state             = model.apply(params, x, mutable=["intermediates"])
            attn = state["intermediates"]["attn_weights"]

            # variable() collections work the same way, e.g. batch_stats:
            y, state = model.apply(
                {"params": params, "batch_stats": stats}, x, mutable=["batch_stats"])
            stats = state["batch_stats"]

        `params` may be either a bare params pytree, or a dict of
        {"params": ..., "<collection>": ..., ...} if you're also feeding in
        existing variable() collections (e.g. batch_stats) to read/update.
        `rngs` optionally supplies named RNG streams (e.g. {"dropout": key})
        for use via self.make_rng("dropout") inside __call__.
        """
        dispatch = self._require_dispatch()
        if "params" in params and set(params.keys()) - {"params"}:
            real_params = params["params"]
            init_variables = {k: v for k, v in params.items() if k != "params"}
        else:
            real_params = params
            init_variables = {}
        root = Scope("apply", real_params, None, name="", owner=self,
                     mutable=frozenset(mutable), rngs=rngs, variables=init_variables)
        token = _push(root)
        try:
            out = dispatch(self, *args, **kwargs)
        finally:
            _pop(token)
        if root.mutable:
            state = {**root.collections, **root.variables}
            return out, state
        return out

    def _require_dispatch(self):
        dispatch = getattr(type(self), "_compact_dispatch", None)
        if dispatch is None:
            raise TypeError(
                f"{type(self).__name__} has no @compact-decorated __call__. "
                f"Define one: \n"
                f"    @nn.compact\n"
                f"    def __call__(self, x): ..."
            )
        return dispatch

    # -- leaf param creation, for use inside a @compact __call__ ----------------

    def param(self, name: str, init_fn: Callable, *shape) -> Any:
        """
        Create (init mode) or fetch (apply mode) a raw learnable array.

        init_fn is called as init_fn(key, *shape) during .init(); during
        .apply() the stored value is returned unchanged and init_fn/shape
        are ignored.
        """
        scope = current_scope()
        if scope is None:
            raise RuntimeError(
                f"{type(self).__name__}.param() called outside a scope — "
                f"only call this from inside a @compact __call__."
            )
        if scope.mode == "init":
            key = scope.next_rng()
            value = init_fn(key, *shape)
            scope.params[name] = value
            return value
        if name not in scope.params:
            raise KeyError(
                f"Missing param '{name}' for {type(self).__name__} "
                f"(scope '{scope.name}') — params pytree doesn't match "
                f"this module's structure."
            )
        return scope.params[name]

    def sow(self, col: str, name: str, value: Any, *,
            init_fn: Callable[[], Any] = tuple,
            reduce_fn: Callable[[Any, Any], Any] = lambda acc, v: acc + (v,)) -> bool:
        """
        Stash a non-learnable intermediate value (e.g. attention weights)
        into a side collection, for use inside a @compact __call__.

        No-op (returns False) unless the caller requested this collection
        via mutable=[col, ...] on .init()/.apply()/.init_with_output() —
        this way sow() calls are always safe to leave in place; they only
        cost anything when someone actually asks for that collection.

            self.sow("intermediates", "attn_weights", attn)
            ...
            y, state = model.apply(params, x, mutable=["intermediates"])
            state["intermediates"]["LinkAttention_0"]["attn_weights"]  # (tuple of) values

        Parameters
        ----------
        col       : collection name, e.g. "intermediates"
        name      : key within this module's slice of that collection
        value     : the value to store
        init_fn   : called with no args to seed the accumulator the first
                    time `name` is sown in this scope. Default: tuple()
        reduce_fn : combines (accumulator, value) -> new accumulator.
                    Default appends to a tuple, so sowing the same name
                    multiple times in one call (e.g. a module invoked in a
                    loop) collects every value instead of overwriting.

        Returns
        -------
        bool — True if stored, False if `col` isn't mutable right now.
        """
        scope = current_scope()
        if scope is None:
            raise RuntimeError(
                f"{type(self).__name__}.sow() called outside a scope — "
                f"only call this from inside a @compact __call__."
            )
        if col not in scope.mutable:
            return False
        bucket = scope.collections.setdefault(col, {})
        acc = bucket[name] if name in bucket else init_fn()
        bucket[name] = reduce_fn(acc, value)
        return True

    def variable(self, col: str, name: str, init_fn: Callable[[], Any]) -> "Variable":
        """
        Create (first call) or fetch a mutable, non-gradient piece of state,
        for use inside a @compact __call__ — e.g. BatchNorm running stats.

        Unlike self.param() (always in the "params" collection, always
        gradient-updated by your optimizer), variable() lives in whatever
        collection name you give it (e.g. "batch_stats") and is read/written
        by your own __call__ logic instead — see Variable.value below.

            ra_mean = self.variable("batch_stats", "mean", lambda: jnp.zeros(dim))
            ra_mean.value = 0.9 * ra_mean.value + 0.1 * batch_mean   # update

        Only actually persists across calls if the collection is requested
        via mutable=[col, ...] on .init()/.apply() — same opt-in convention
        as sow(). If `col` isn't mutable right now, this still works for a
        single call (falls back to a fresh, unshared value each time) so
        it's safe to leave variable() calls in place regardless of caller.

        Returns
        -------
        Variable — a thin box with a .value property; read/write it as
        many times as you like inside this __call__.
        """
        scope = current_scope()
        if scope is None:
            raise RuntimeError(
                f"{type(self).__name__}.variable() called outside a scope — "
                f"only call this from inside a @compact __call__."
            )
        bucket = scope.variables.setdefault(col, {})
        if name not in bucket:
            bucket[name] = init_fn()
        return Variable(scope, col, name)

    def make_rng(self, name: str = "params"):
        """
        Split off a fresh PRNGKey from the named RNG stream `name` (e.g.
        "dropout"), for use inside a @compact __call__ — flax-style named
        RNG streams, separate from the key used to build params.

        Requires that stream to have been supplied via rngs={name: key, ...}
        to .init()/.apply()/.init_with_output().

            key = self.make_rng("dropout")
            x = F.dropout(x, rate, key, deterministic=deterministic)
        """
        scope = current_scope()
        if scope is None:
            raise RuntimeError(
                f"{type(self).__name__}.make_rng() called outside a scope — "
                f"only call this from inside a @compact __call__."
            )
        return scope.next_named_rng(name)

    def is_initializing(self) -> bool:
        """True if currently running inside .init()/.init_with_output(), False if .apply()."""
        scope = current_scope()
        if scope is None:
            raise RuntimeError(
                f"{type(self).__name__}.is_initializing() called outside a "
                f"scope — only call this from inside a @compact __call__."
            )
        return scope.mode == "init"

    def bind(self, params: Dict[str, Any]) -> "BoundModule":
        """
        Return a lightweight callable bound to `params`, for interactive/
        notebook use — no explicit .apply(params, ...) each time:

            bound = model.bind(params)
            y1 = bound(x1)
            y2 = bound(x2)

        Purely functional convenience (unlike @pack, nothing is cached or
        mutated on `self`) — every call is exactly model.apply(params, ...).
        For training, still extract params and call .apply()/.init()
        directly so jax.grad has explicit arguments to differentiate.
        """
        return BoundModule(self, params)

    def copy(self, **overrides) -> "Module":
        """
        Return a new instance of this module's class with the same config,
        optionally overriding some fields — flax.linen-style .copy() (a thin
        wrapper over dataclasses.replace, since every Module is a dataclass):

            wide = attn.copy(num_heads=16)

        Works for any dataclass-based Module (i.e. anything that didn't opt
        out of auto-dataclassing by writing its own __init__). If you wrote
        a custom __init__, override copy() yourself.
        """
        if not dataclasses.is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} isn't a dataclass (it defines its "
                f"own __init__), so the default .copy() can't introspect its "
                f"fields — override .copy() on that class."
            )
        return dataclasses.replace(self, **overrides)

    def tabulate(self, key, *args, **kwargs) -> str:
        """
        Return a plain-text summary table of this module's param tree:
        one row per leaf param (dotted path, shape, dtype, count), plus a
        total. Uses jax.eval_shape so no real compute/memory is used.

            print(model.tabulate(key, x))

        No 'rich' dependency (unlike flax.linen.tabulate) — this package
        stays dependency-free by design (see project notes); output is a
        plain monospace string, safe to print anywhere.
        """
        shapes = jax.eval_shape(lambda k: self.init(k, *args, **kwargs), key)

        rows: list[Tuple[str, str, str, int]] = []

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, path + [k])
            else:
                count = 1
                for d in node.shape:
                    count *= d
                rows.append((".".join(path), str(tuple(node.shape)), str(node.dtype), count))

        walk(shapes, [])

        path_w = max([len("path")] + [len(r[0]) for r in rows])
        shape_w = max([len("shape")] + [len(r[1]) for r in rows])
        dtype_w = max([len("dtype")] + [len(r[2]) for r in rows])
        count_w = max([len("params")] + [len(f"{r[3]:,}") for r in rows])

        def fmt_row(a, b, c, d):
            return f"{a:<{path_w}}  {b:<{shape_w}}  {c:<{dtype_w}}  {d:>{count_w}}"

        header = fmt_row("path", "shape", "dtype", "params")
        sep = "-" * len(header)
        lines = [f"{type(self).__name__} — tabulate", sep, header, sep]
        for path, shape, dtype, count in rows:
            lines.append(fmt_row(path, shape, dtype, f"{count:,}"))
        total = sum(r[3] for r in rows)
        lines.append(sep)
        lines.append(fmt_row("TOTAL", "", "", f"{total:,}"))
        return "\n".join(lines)


class Variable:
    """
    Thin read-write box returned by Module.variable(), e.g.:

        ra_mean = self.variable("batch_stats", "mean", lambda: jnp.zeros(dim))
        current = ra_mean.value
        ra_mean.value = 0.9 * current + 0.1 * new_mean

    Reads/writes go straight through to this scope's slice of
    scope.variables[col][name] — there's no separate copy to keep in sync.
    """

    __slots__ = ("_scope", "_col", "_name")

    def __init__(self, scope: Scope, col: str, name: str):
        self._scope = scope
        self._col = col
        self._name = name

    @property
    def value(self) -> Any:
        return self._scope.variables[self._col][self._name]

    @value.setter
    def value(self, new_value: Any) -> None:
        self._scope.variables[self._col][self._name] = new_value

    def __repr__(self) -> str:
        return f"<Variable {self._col}.{self._name} = {self.value!r}>"


class BoundModule:
    """
    A Module + its params, returned by Module.bind(). Calling it runs
    module.apply(params, *args, **kwargs). See Module.bind() for details.
    """

    __slots__ = ("_module", "_params")

    def __init__(self, module: "Module", params: Dict[str, Any]):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_params", params)

    def __call__(self, *args, **kwargs):
        return self._module.apply(self._params, *args, **kwargs)

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def __repr__(self) -> str:
        return f"<BoundModule {type(self._module).__name__}>"


def _make_setup_dispatch(raw_call: Callable) -> Callable:
    """
    Build the dispatch used for setup()-style Modules (as opposed to
    @compact). Runs self.setup() once, then the plain __call__ body, both
    inside whatever scope is already active for `self` — mirroring
    @compact's behavior when `parent.owner is self` (the root/self case),
    since setup() is only ever the entry point for a module invoked as a
    submodule, never called a second time re-entrantly the way @compact
    submodules can be invoked repeatedly inside a loop.

    Submodules assigned in setup() (e.g. self.dense1 = Linear(...)) get
    their own child scope the FIRST TIME THEY'RE CALLED, exactly like any
    other nested @compact submodule invocation — setup() itself doesn't
    need special scope handling, it just needs to run before __call__.
    """

    @functools.wraps(raw_call)
    def dispatch(self, *args, **kwargs):
        # Idempotent: if this exact instance is reused across multiple
        # init()/apply() passes, setup() should still only configure
        # attributes freshly each time (cheap: it just assigns submodule
        # instances, no heavy compute), so no guard needed — re-running is
        # harmless and keeps behavior correct if params differ per call.
        self.setup()
        return raw_call(self, *args, **kwargs)

    dispatch._is_compact = False
    return dispatch


# ---------------------------------------------------------------------------
# @compact
# ---------------------------------------------------------------------------

def compact(method: Callable) -> Callable:
    """
    Decorate a Module's __call__ so it can declare submodules inline.

    Handles two cases:
    - Root call (via .init()/.apply()): runs `method` directly against the
      scope those already pushed — no extra nesting.
    - Nested call (module invoked from inside ANOTHER module's @compact
      __call__): creates a child Scope addressed by name (explicit
      name="..." or auto-numbered "ClassName_i" by call order), runs
      `method` inside it, and — during .init() — writes the child's
      finished param dict back into the parent under that name.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        parent = current_scope()
        if parent is None:
            raise RuntimeError(
                f"{type(self).__name__}() called with no active scope. "
                f"Use .init(key, ...) or .apply(params, ...) to run a "
                f"top-level module; only call submodules from inside "
                f"another module's @compact __call__."
            )
        if parent.owner is self:
            # We ARE the scope .init()/.apply() just pushed for this exact
            # instance — run directly, no new child scope needed.
            return method(self, *args, **kwargs)

        # Nested submodule: get/create our own slice of the param tree.
        name = self._name or parent.auto_name(type(self).__name__)
        if parent.mode == "init":
            child_params: Dict[str, Any] = {}
            child_rng = parent.next_rng()
        else:
            if name not in parent.params:
                raise KeyError(
                    f"Missing params for submodule '{name}' "
                    f"({type(self).__name__}) under scope '{parent.name}' — "
                    f"params pytree doesn't match this module's structure."
                )
            child_params = parent.params[name]
            child_rng = None

        # Slice any pre-existing variable() collections (e.g. batch_stats
        # fed into .apply()) down to this submodule's key, same idea as
        # child_params above.
        child_variables: Dict[str, Dict[str, Any]] = {}
        for col, bucket in parent.variables.items():
            if name in bucket:
                child_variables[col] = bucket[name]

        child = Scope(parent.mode, child_params, child_rng, name=name, owner=self,
                      mutable=parent.mutable, rngs=parent.rngs, variables=child_variables)
        token = _push(child)
        try:
            out = method(self, *args, **kwargs)
        finally:
            _pop(token)
        if parent.mode == "init":
            parent.params[name] = child.params
        # Merge sown collections back up regardless of mode — sow() is
        # normally used during .apply(), unlike params which only get
        # created during .init().
        for col, bucket in child.collections.items():
            if bucket:
                parent.collections.setdefault(col, {})[name] = bucket
        # Merge variable() collections back up the same way (read-write,
        # so always merge back regardless of whether values changed).
        for col, bucket in child.variables.items():
            parent.variables.setdefault(col, {})[name] = bucket
        return out

    wrapper._is_compact = True
    return wrapper


# ---------------------------------------------------------------------------
# @pack
# ---------------------------------------------------------------------------

def pack(cls=None, *, seed: int = 0):
    """
    Class decorator: opt-in PyTorch/Equinox-style stateful calling.

        @nn.pack
        class MyModel(nn.Module):
            d: int
            @nn.compact
            def __call__(self, x): ...

        model = MyModel(d=512)
        out   = model(x)      # auto-inits on first call (seed=0 by default,
                               # or pass seed=... to @nn.pack(seed=...)),
                               # reuses stored params on every call after

    Also usable directly: `PackedModel = nn.pack(MyModel)`.

    This overwrites __call__ for convenience; it does NOT replace the
    functional API — .init() and .apply() (and .param() inside nested
    submodules) still work exactly as before via Module._compact_dispatch,
    which @pack never touches. For training, prefer the functional form:

        params = model.params                      # extract
        grads  = jax.grad(loss_fn)(params, x, y)    # loss_fn calls model.apply(params, ...)
        model.params = optimizer_step(params, grads) # write back

    since jax.grad only differentiates through explicit function arguments,
    not through attributes read off a Python object.
    """
    def decorator(cls):
        def packed_call(self, *args, **kwargs):
            if getattr(self, "_packed_params", None) is None:
                key = jax.random.key(getattr(self, "_pack_seed", seed))
                self._packed_params = self.init(key, *args, **kwargs)
            return self.apply(self._packed_params, *args, **kwargs)

        cls.__call__ = packed_call
        cls._pack_seed = seed

        def _get_params(self):
            return self._packed_params

        def _set_params(self, value):
            self._packed_params = value

        cls.params = property(_get_params, _set_params)
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator
