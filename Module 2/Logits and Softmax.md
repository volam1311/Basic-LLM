# Logits and softmax

The LM head turns the last token's embedding into one score per vocabulary word. Softmax turns those scores into probabilities.

`x` is the last token's embedding, shape `(d_model,)`. `W_lm` has shape `(d_model, vocab)`.

```text
logits = x @ W_lm
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

The logits are raw scores. They can be negative and they do not sum to 1. Softmax makes every entry positive, and the entries sum to 1.

This example uses five words so the arithmetic fits on the page: `mat`, `floor`, `table`, `banana`, `hat`, with scores `[2, 1, 0, -1, -2]`.

```python
import numpy as np

words = ["mat", "floor", "table", "banana", "hat"]
logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0])

def softmax(z):
    z = z - z.max()          # subtract the max so exp stays finite
    e = np.exp(z)
    return e / e.sum()

probs = softmax(logits)
print(np.round(probs, 4))    # [0.6364 0.2341 0.0861 0.0317 0.0117]
print(probs.sum())           # 1.0
```

`mat` gets about 0.636, `floor` about 0.234, `table` about 0.086, `banana` about 0.032, and `hat` about 0.012. Subtracting the max inside `softmax` does not change the result. `exp` of a large logit would overflow; the max shift prevents that.

Next: [The last-word predictor](<The Last-Word Predictor.md>)
