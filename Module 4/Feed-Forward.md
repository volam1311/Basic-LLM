# The feed-forward network

Attention mixes tokens. The feed-forward network does not. It applies the same two linear layers to every token on its own. A stack of linear layers with no activation is still one linear layer, so the nonlinearity is what makes an extra block a new computation. GELU is the activation used here. ReLU is the older alternative, which replaces every negative entry with zero.

```text
expand = x @ W_1 + b_1      d_model → d_ff
hidden = GELU(expand)
output = hidden @ W_2 + b_2  d_ff → d_model
```

In the paper, `d_ff` is four times `d_model`. The snippet uses `d_model = 2` and `d_ff = 4`, and the tanh approximation of GELU.

```python
import numpy as np

def gelu(v):
    return 0.5 * v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v**3)))

W_1 = np.array([[1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0]])
W_2 = np.array([[1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0]])

def feed_forward(x):
    hidden = gelu(x @ W_1)
    return hidden @ W_2

x = np.array([1.0, 0.0])
print(np.round(feed_forward(x), 4))       # [1.6824 0.    ]
print(np.round(feed_forward([0.0, 1.0]), 4))  # [0.     1.6824]
```

`GELU(1)` is about 0.841, and `GELU(0)` is 0. The first token expands to `[1, 1, 0, 0]`, GELU keeps about `[0.841, 0.841, 0, 0]`, and `W_2` compresses that to about `[1.682, 0]`. The second token is transformed by the same weights and never reads the first token.

Without GELU, two linear maps collapse into one. `(x @ A) @ B` equals `x @ (A @ B)`.

```python
import numpy as np

A = np.array([[1.0, 2.0], [0.0, 1.0]])
B = np.array([[2.0, 0.0], [1.0, 3.0]])
x = np.array([0.5, -1.0])
print((x @ A) @ B)     # [1. 0.]
print(x @ (A @ B))     # [1. 0.]
```

Next: [One block, then a stack](<One Block.md>)
