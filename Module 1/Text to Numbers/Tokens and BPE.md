# Tokens and BPE

A model cannot take raw characters as "a word" until the text has been split into pieces it has an id for. Those pieces are tokens.

## Whole words

A word-level vocabulary gives each known word an integer.

```python
vocab = {"the": 0, "big": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}

def word_ids(text):
    return [vocab[word] for word in text.split()]

print(word_ids("the big cat sat on the mat"))
# [0, 1, 2, 3, 4, 0, 5]
```

`word_ids("the big cats")` raises `KeyError`. "cats" was never stored, so it has no id. A vocabulary that tried to store every word would also be enormous, and it would still miss the next new word.

The lessons after this one use a small word-level vocabulary of about this size, so the examples stay readable. A full model uses subwords instead.

## One BPE merge

Byte-pair encoding starts from characters and merges the pair that shows up most often. Frequent chunks become single tokens. This course does not train that procedure. One result is enough to see why it helps.

Take the characters of "lower", in a vocabulary that has already merged the frequent chunks `low` and `er`:

```text
l o w e r   →   low er
```

"lower" is two tokens the model already knows, `low` and `er`, even if the whole word is rare. "cats" can be `cat` plus `s` for the same reason. Unknown text falls back to smaller pieces instead of a missing id.

Next: [Embeddings](Embeddings.md)
