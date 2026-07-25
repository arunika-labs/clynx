# clynx

**SLAM** (Super Link Attention Mechanism) + **Mulion** optimizer, in pure JAX.

```
pip install git+https://github.com/arunika-labs/clynx.git
```

GPU/TPU variants:

```
pip install "clynx[cuda12] @ git+https://github.com/arunika-labs/clynx.git"
pip install "clynx[cuda13] @ git+https://github.com/arunika-labs/clynx.git"
pip install "clynx[tpu]    @ git+https://github.com/arunika-labs/clynx.git"
```

---

## SLAM — Link Attention

Replaces softmax attention with sigmoid:

```
LinkAttention(Q, K, V) = sigmoid(QKᵀ / τ) · V
```

τ is learnable per head, initialized to d_k (not √d_k — see rationale in
`clynx/slam/modules.py`). Normalize with plain LayerNorm (bring your own;
SymLogNorm was tried and dropped after experimentation).

```python
import clynx.slam as nn
import jax

model  = nn.LinkAttention(num_heads=8, d_model=512)
params = model.init(jax.random.key(0), x)
y      = model.apply(params, x)          # (batch, seq, 512)
```

Available modules: `Linear`, `Embed`, `LinkAttention`, `FeedForward`, `Dropout`.

---

## Mulion — optimizer

Muon (2D+ params) + Lion (1D params), unified. No external optimizer library.

```python
from clynx.optim import mulion
import jax

opt       = mulion(lr=1e-3, momentum=0.95, ns_steps=5)
opt_state = opt.init(params)

# training step
grads = jax.grad(loss_fn)(params)
updates, opt_state = opt.update(grads, opt_state, params)
params = jax.tree.map(lambda p, u: p + u, params, updates)
```

Memory: **6 bytes/param** (vs Adam's 10) — one fp32 momentum buffer, no second moment.

| Param ndim | Branch | Operation              |
|------------|--------|------------------------|
| ≥ 2        | Muon   | Newton-Schulz ortho    |
| 1          | Lion   | sign(momentum)         |

Set `ns_steps=0` to degrade Muon → SGD momentum (ablation).

---

## SymLog gradient compression

```python
import clynx.slam.functional as F

# Use symlog_stable (not symlog) — has custom VJP that compresses gradients
x = F.symlog_stable(x)
```

The custom backward pass applies SymLog to incoming gradients instead of the
standard `1/(1+|x|)` derivative — prevents vanishing across stacked layers
while preserving sign and relative order.

Compatible optimizers: **Muon ✅  Lion ✅  SGD ⚠️  Adam ❌**

---

## Requirements

- Python ≥ 3.12
- JAX 0.11.0
