# Vectors and the dot product

A transformer stores each token as a list of numbers. In code that list is an array. In math it is a vector. Snippets in this course use NumPy (`pip install -r requirements.txt` from the repo root).

```python
import numpy as np

cat = np.array([0.2, 0.9, 0.1])
dog = np.array([0.3, 0.8, 0.2])
mat = np.array([0.9, 0.1, 0.4])

print(2 * cat)    # [0.4 1.8 0.2]
print(cat @ dog)  # 0.8
print(cat @ mat)  # 0.31
```

Scaling a vector multiplies every entry by the same number, as in `2 * cat`.

The operation attention uses is the dot product. Multiply matching entries, then add.

`cat · dog = 0.2*0.3 + 0.9*0.8 + 0.1*0.2 = 0.80`. `cat · mat = 0.31`. `cat` is more aligned with `dog` than with `mat`.

In Module 3, the attention score between two tokens is this kind of dot product, divided by the square root of the key dimension.

Next: [Matrices, shapes, and multiplication](<Matrices and Shapes.md>)
