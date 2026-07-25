"""
clynx.serialize
~~~~~~~~~~~~~~~
Serialization utilities for clynx pytrees.

Submodules
----------
    safetensors     Save/load JAX pytrees via the safetensors format.
                    Handles flatten/unflatten automatically using dot-separated keys.

Usage
-----
    from clynx.serialize import safetensors as st

    # save
    st.save(state.params, "checkpoints/step_1000.safetensors")

    # load back into the same structure
    params = st.load("checkpoints/step_1000.safetensors", target=params)

    # or inspect as flat dict
    flat = st.load("checkpoints/step_1000.safetensors")
    print(sorted(flat.keys()))
"""

from clynx.serialize import safetensors

__all__ = ["safetensors"]
