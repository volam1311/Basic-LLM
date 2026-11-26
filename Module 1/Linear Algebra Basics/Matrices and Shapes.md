# Matrices, shapes, and multiplication

A matrix is a two-dimensional array. Its shape is `(rows, columns)`.

An embedding table for 20 tokens, with 4 numbers each, has shape `(20, 4)`. Token 15 is row 15, a vector of length 4.

Two arrays can multiply when the inner dimensions match. The result takes the outer dimensions:

```text
(rows, shared) @ (shared, cols) → (rows, cols)
```

A small multiply shows the rule. Each entry of the result is a dot product of one row from the left and one column from the right.

```python
import numpy as np

a = np.array([[1.0, 2.0]])          # (1, 2)
b = np.array([[1.0, 0.0, 3.0],      # (2, 3)
              [0.0, 1.0, 4.0]])
print(a @ b)                        # [[1. 2. 11.]]  shape (1, 3)
```

`1*1 + 2*0 = 1`, `1*0 + 2*1 = 2`, `1*3 + 2*4 = 11`.

The same check is the LM head in Module 2. A last-token vector of shape `(1, 4)` times `W_lm` of shape `(4, 20)` gives shape `(1, 20)`: one score for each of 20 vocabulary words.

It is also how attention builds its three matrices. A matrix of token vectors times `W_Q` is the query matrix. `W_K` and `W_V` build the keys and the values. When a layer fails, the first check is whether those inner dimensions agree.

Next: [Tokens and BPE](<../Text to Numbers/Tokens and BPE.md>)
