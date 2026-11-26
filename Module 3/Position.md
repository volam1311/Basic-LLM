# Position

The score `Q @ K.T` is a dot product of vectors. The formula has no index for "first" or "second". Two tokens that share an embedding are interchangeable: swapping them does not change `X`, so it does not change the scores.

```python
import numpy as np

X = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
])
print(X @ X.T)
# [[1. 1.]
#  [1. 1.]]

P = np.array([
    [0.0, 0.0, 1.0, 0.0],  # position 0
    [0.0, 0.0, 0.0, 1.0],  # position 1
])
print((X + P) @ (X + P).T)
# [[2. 1.]
#  [1. 2.]]
```

Add a position vector of length `d_model` to each token embedding before the block. The two rows were equal. After the add, slot 0 and slot 1 carry different vectors, and the scores change.

Two versions of that vector are common. A learned encoding stores one vector per index and trains it with the rest of the weights. The paper uses a sinusoidal encoding, a fixed pattern of sines and cosines of the index. This course uses the add itself. It does not derive the sine formula.

Next: [A context-aware predictor](<Context-Aware Predictor.md>)
