# Multi-head attention

The tiny picture of this course uses 4 heads. Each head runs the same formula as the [one-head lesson](<One Attention Head.md>), with its own `W_Q`, `W_K`, and `W_V`. The key dimension is smaller: `d_k = d_model / n_heads`.

With `d_model = 8` and 4 heads, each head has `d_k = 2`. A head's output is `(n_tokens, 2)`. Concatenate the four outputs along the feature axis and the matrix is `(n_tokens, 8)` again. `W_O`, shape `(d_model, d_model)`, mixes the heads back into one vector per token.

```python
import numpy as np

n_tokens, d_model, n_heads = 7, 8, 4
d_k = d_model // n_heads  # 2

# Stand-in outputs. A real head would compute weights @ V, shape (n_tokens, d_k).
heads = [np.zeros((n_tokens, d_k)) for _ in range(n_heads)]
joined = np.concatenate(heads, axis=1)          # (7, 8)
W_O = np.eye(d_model)                           # (8, 8)
output = joined @ W_O                           # (7, 8)
print(output.shape)
```

Heads can learn different lookups. On `the big cat sat on the big ___`, one head can put most of its weight on the earlier `big` when the current token is the second `big`. The row below is an example of that pattern, not a row from a trained model. It sums to 1.

```text
tokens   the   big   cat   sat   on    the   big
weight   0.04  0.60  0.04  0.04  0.04  0.04  0.20
```

The current token keeps some weight on itself (0.20). The largest weight, 0.60, sits on the earlier `big`. Another head can put its weight somewhere else. After `W_O`, each token still has a single vector of length `d_model`.

Next: [Position](Position.md)
