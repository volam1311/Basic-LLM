# Training vs Inference:

We know the transformer architecture: embeddings flow through and feed-forward blocks. then the LM head produces the prediction. The flow from the input to output is called forward pass.

Every time we ask ChatGPT a question, it runs this forward pass. The text goes in and the predictions come out. The weights stay frozen, they donot change. This is called the inference mode: where we use a pretrained model to make the predictions.

## Where the weights come from?

The weight were not always useful. Before training, they are random numbers. A random transformer produces garbage as it might predict the wrong next word. 

Training is how those random weights become useful pattern detectors. The model processes billions of text examples, compares its prediction to the next actual words, and adjusts its weights to make better predictions. 

## Training adds Backward Pass:

Training needs more than the forward pass. After each prediction, it must figure out how to improve. The backward pass (backpropagation) computes the gradients. This requires storing immediate values from the forward pass and running calculation backward through every layer.

Inference skips all of this. Weights are frozen, the model just run forward and output the prediction.

## Understanding Forward Pass:

Understanding the forward pass is the foundation. It is where you learn what each component actually does:

- How embeddings convert words to vectors
- How attention finds relationships between words
- How feed-forward networks transform representations
- How the LM head produces vocabulary scores
- How softmax converts scores to probabilities

Training adds some more on top of the forward pass (optimization, loss functions, gradients) but the core computation is the same. 

## Pre-trained Tiny Model:

We will work with a tiny model that has been trained. The weights are frozen and we are purely in the inference mode. We will load the weights, run forward pass, and see how predictions emerge from the architecture.