# Basic LLM

How a transformer turns text into the next word. Four modules, in order: text becomes vectors, a last token becomes a probability, attention reads the whole prefix, and a full block generates text. The notes follow the forward pass. They do not train a model.

The running example, introduced in Module 2, is `the big cat sat on the big ___`.

Snippets use NumPy.

```bash
pip install -r requirements.txt
```

## Module 1 — From text to numbers

- [Introduction](Module%201/Introduction.md)
- [What an LLM does](Module%201/Fundamentals/What%20an%20LLM%20Does.md)
- [The pipeline](Module%201/Fundamentals/Pipeline.md)
- [Training and inference](Module%201/Fundamentals/Training%20vs%20Inference.md)
- [Vectors and the dot product](Module%201/Linear%20Algebra%20Basics/Vectors%20and%20the%20Dot%20Product.md)
- [Matrices, shapes, and multiplication](Module%201/Linear%20Algebra%20Basics/Matrices%20and%20Shapes.md)
- [Tokens and BPE](Module%201/Text%20to%20Numbers/Tokens%20and%20BPE.md)
- [Embeddings](Module%201/Text%20to%20Numbers/Embeddings.md)

## Module 2 — Predicting the next word

- [Logits and softmax](Module%202/Logits%20and%20Softmax.md)
- [The last-word predictor](Module%202/The%20Last-Word%20Predictor.md)

## Module 3 — Attention

- [Why the last token is not enough](Module%203/Why%20the%20Last%20Token%20Is%20Not%20Enough.md)
- [One attention head](Module%203/One%20Attention%20Head.md)
- [Multi-head attention](Module%203/Multi-Head%20Attention.md)
- [Position](Module%203/Position.md)
- [A context-aware predictor](Module%203/Context-Aware%20Predictor.md)

## Module 4 — The transformer

- [Residuals and layer norm](Module%204/Residuals%20and%20Layer%20Norm.md)
- [The feed-forward network](Module%204/Feed-Forward.md)
- [One block, then a stack](Module%204/One%20Block.md)
- [Generation](Module%204/Generation.md)
- [Paper map](Module%204/Paper%20Map.md)
