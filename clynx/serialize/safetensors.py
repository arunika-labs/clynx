"""
clynx.serialize.safetensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Save and load JAX pytrees using the safetensors format.

The core problem: safetensors stores a flat dict of str -> tensor.
A clynx params pytree is a nested dict, e.g.:

    {
        "embed":    {"E": Array},
        "blocks.0": {"attn": {"q_proj": {"W": Array, "b": Array}, ...}},
        "blocks.1": {"attn": {"q_proj": {"W": Array, "b": Array}, ...}},
    }

This module flattens that tree to dot-separated keys before saving and
reconstructs it on load — the safetensors file itself only ever sees
flat string keys, which is what it expects.

    # flat form:
    {
        "embed.E":                    Array,
        "blocks.0.attn.q_proj.W":     Array,
        "blocks.0.attn.q_proj.b":     Array,
        ...
    }

Usage
-----
    from clynx.serialize import safetensors as st

    st.save(params, "model.safetensors")
    params = st.load("model.safetensors", target=params)

    # or load without a target (returns flat dict of jax arrays):
    flat = st.load("model.safetensors")

API
---
    save(pytree, path)         flatten + write .safetensors file
    load(path, target=None)    read file; if target given, unflatten into
                               matching pytree structure; else return flat dict
    flatten(pytree)            pytree -> flat {dot.key: array} dict
    unflatten(flat, target)    flat {dot.key: array} -> pytree like target
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from safetensors.numpy import save_file, load_file
except ImportError as e:
    raise ImportError(
        "safetensors is required for clynx.serialize.safetensors. "
        "Install it with:  pip install safetensors"
    ) from e


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------

def flatten(pytree: Any, sep: str = ".") -> Dict[str, np.ndarray]:
    """
    Flatten a nested pytree of JAX arrays to a dot-separated dict.

    Parameters
    ----------
    pytree : Any
        Nested dict (or any JAX-registered pytree) whose leaves are Arrays.
    sep : str
        Separator for nested keys. Default ".".
        Keys that already contain sep are left as-is — just avoid using "."
        in your own key names to keep round-trips unambiguous.

    Returns
    -------
    Dict[str, np.ndarray]
        Flat mapping suitable for safetensors.numpy.save_file.

    Example
    -------
    >>> flatten({"a": {"b": jnp.ones(3)}})
    {"a.b": array([1., 1., 1.])}
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree)
    flat: Dict[str, np.ndarray] = {}
    for key_path, leaf in leaves:
        # Build dot-separated key from the JAX key path
        parts = []
        for k in key_path:
            # DictKey, SequenceKey, etc. all have a __str__ that includes
            # brackets; we strip those to get a clean dot-separated key.
            s = str(k)
            # jax key path elements render as "[key]" for dicts and "[n]" for lists
            s = s.strip("[]'\"")
            parts.append(s)
        dot_key = sep.join(parts)
        flat[dot_key] = np.array(leaf)
    return flat


def unflatten(flat: Dict[str, Any], target: Any, sep: str = ".") -> Any:
    """
    Reconstruct a pytree from a flat dot-separated dict, using target as
    the structural template.

    Parameters
    ----------
    flat   : Dict[str, array-like]
        Flat mapping as produced by flatten() or loaded from safetensors.
    target : Any
        A pytree with the same structure as the one that was originally
        flattened. Only the structure (treedef) is used — leaf values are
        replaced by the arrays from flat.
    sep    : str
        Key separator used when the file was saved. Default ".".

    Returns
    -------
    Pytree matching target's structure with leaves filled from flat.

    Raises
    ------
    KeyError
        If a key expected by target's treedef is not found in flat.
    """
    leaves_target, treedef = jax.tree_util.tree_flatten_with_path(target)

    new_leaves = []
    for key_path, _ in leaves_target:
        parts = []
        for k in key_path:
            s = str(k).strip("[]'\"")
            parts.append(s)
        dot_key = sep.join(parts)
        if dot_key not in flat:
            raise KeyError(
                f"Key '{dot_key}' not found in checkpoint. "
                f"Available keys: {sorted(flat.keys())[:10]}{'...' if len(flat) > 10 else ''}"
            )
        new_leaves.append(jnp.array(flat[dot_key]))

    return treedef.unflatten(new_leaves)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save(pytree: Any, path: str | Path) -> None:
    """
    Save a JAX pytree to a safetensors file.

    Parameters
    ----------
    pytree : Any
        Nested pytree of JAX arrays (params, TrainState.params, etc.).
    path   : str or Path
        Destination file path. Conventionally ends in ".safetensors".

    Example
    -------
        from clynx.serialize import safetensors as st
        st.save(state.params, "checkpoints/step_1000.safetensors")
    """
    flat = flatten(pytree)
    save_file(flat, str(path))


def load(
    path: str | Path,
    target: Optional[Any] = None,
    sep: str = ".",
) -> Any:
    """
    Load a safetensors file into a JAX pytree.

    Parameters
    ----------
    path   : str or Path
        Path to a .safetensors file saved with save().
    target : Any or None
        If given, unflatten the loaded tensors into this pytree's structure.
        Typically the initial params pytree from model.init().
        If None, returns a flat dict of {dot.key: jnp.array}.
    sep    : str
        Key separator used when the file was saved. Default ".".

    Returns
    -------
    If target is provided: pytree matching target's structure.
    If target is None: flat Dict[str, jnp.Array].

    Example
    -------
        # with target (recommended):
        params = st.load("step_1000.safetensors", target=params)

        # without target (inspect raw tensors):
        flat = st.load("step_1000.safetensors")
        print(flat.keys())
    """
    raw = load_file(str(path))          # Dict[str, np.ndarray]

    if target is None:
        return {k: jnp.array(v) for k, v in raw.items()}

    return unflatten(raw, target, sep=sep)
