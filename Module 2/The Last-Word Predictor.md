# The last-word predictor

Argmax picks the largest probability. On the five-word distribution from the previous lesson, the largest entry is `mat` at about 0.636. Appending it produces "The cat sat on the mat".

```python
import numpy as np

words = ["mat", "floor", "table", "banana", "hat"]
logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
choice = words[int(logits.argmax())]
print(choice)  # mat
```

Argmax reads the scores directly. Softmax does not change which entry wins, because it preserves order.

## The blank this predictor cannot fill

The running example for the rest of the course is:

```text
the big cat sat on the big ___
```

A last-word predictor builds `x` from the last token only, then computes `logits = x @ W_lm`. The last token is `big`. The earlier `big`, and the `cat`, never enter the formula.

The weights below are chosen so the arithmetic is easy. They are not a trained model. `big` selects row 0 of `W_lm`. Any prefix that ends in `big` selects that same row.

```python
import numpy as np

words = ["the", "big", "cat", "sat", "on", "mat"]
E = np.zeros((len(words), 4))
E[words.index("big")] = [1.0, 0.0, 0.0, 0.0]
E[words.index("cat")] = [0.0, 1.0, 0.0, 0.0]

W_lm = np.zeros((4, len(words)))
W_lm[0] = [0.0, 0.0, 0.5, 0.0, 0.0, 2.0]    # scores used when the last token is big
W_lm[1] = [1.5, 0.2, 0.0, 0.0, 0.0, -1.0]   # scores used when the last token is cat

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def predict(token_ids):
    logits = E[token_ids[-1]] @ W_lm
    probs = softmax(logits)
    return words[int(probs.argmax())], np.round(probs, 4)

ends_in_big = [0, 1, 2, 3, 4, 0, 1]   # the big cat sat on the big
also_big = [0, 2, 3, 4, 0, 1]         # the cat sat on the big
ends_in_cat = [0, 1, 2]               # the big cat

print(predict(ends_in_big))
# ('mat', array([0.0767, 0.0767, 0.1265, 0.0767, 0.0767, 0.5667]))
print(predict(also_big))
# the same distribution
print(predict(ends_in_cat))
# ('the', array([0.4941, 0.1346, 0.1102, 0.1102, 0.1102, 0.0406]))
```

Both prefixes that end in `big` produce the same distribution, peaked on `mat` at about 0.567. The prefix that ends in `cat` produces a different one. The earlier tokens are sitting in the id list, and the formula never reads them.

Next: [Why the last token is not enough](<../Module 3/Why the Last Token Is Not Enough.md>)
