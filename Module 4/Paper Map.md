# Paper map

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) is the source of the blocks in this course. The figure below is the paper's architecture. The left half is the encoder. The right half is the decoder. This course follows the decoder and predicts the next token. It drops the encoder, and it drops the decoder sublayer that attends to the encoder. The block in [One block, then a stack](<One Block.md>) is that decoder block without the cross-attention sublayer.

The figure draws Add & Norm after each sublayer, which is the paper's order. The block lesson uses the order common in later decoder models: layer norm, then the sublayer, then the add. The two pieces are the same.

![Figure 1 from Attention Is All You Need](<../Module 1/img/transformer-paper-figure.png>)

| Paper section | Lesson |
| --- | --- |
| 3.4 Embeddings and Softmax | [Embeddings](<../Module 1/Text to Numbers/Embeddings.md>), [Logits and softmax](<../Module 2/Logits and Softmax.md>) |
| 3.5 Positional Encoding | [Position](<../Module 3/Position.md>) |
| 3.2 Attention | [One attention head](<../Module 3/One Attention Head.md>), [Multi-head attention](<../Module 3/Multi-Head Attention.md>) |
| 3.3 Position-wise Feed-Forward Networks | [The feed-forward network](<Feed-Forward.md>) |
| 3.1 Encoder and Decoder Stacks | [Residuals and layer norm](<Residuals and Layer Norm.md>), [One block, then a stack](<One Block.md>) |
| 5 Training | [Training and inference](<../Module 1/Fundamentals/Training vs Inference.md>). This course does not cover the training setup. |
