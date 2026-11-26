# Embeddings

A token id is a row index. The embedding matrix has one row per token, and each row is a vector of length `d_model`. Token 2 selects row 2.

The rows below are the same three vectors as in the dot-product lesson. `cat` is id 2.

```python
import numpy as np

# rows: the, big, cat, dog, sat, on, mat
E = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.2, 0.9, 0.1],  # cat
    [0.3, 0.8, 0.2],  # dog
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.9, 0.1, 0.4],  # mat
])

cat = E[2]
dog = E[3]
mat = E[6]
print(cat @ dog)  # 0.80
print(cat @ mat)  # 0.31
```

`cat · dog` is larger than `cat · mat`. "Nearby" means a larger dot product between rows. Training is what moves those rows. At inference the matrix is frozen, and the lookup is the whole step.

Next: [Logits and softmax](<../../Module 2/Logits and Softmax.md>)
