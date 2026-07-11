# LLM:

## Text In, Text Out:

When we type "The capital of Viet Nam" into ChatGPT, Gemini and it responds with "Ha Noi". Whenever, we type a question, the LLM answers. These models take the text as input and produce text as output. 

This behavior looks like intelligence, like the model understans what you are asking. However, understanding how LLM works reveal something simpler and more fascinating: the model is predicting the next most likely word, over and over again.

## Predicting the next word:

An LLM's core function is next-word prediction. Given the text "The cat sat on the", the model predicts the output probabilities for what comes next:

- "mat" (high prob)
- "floor" (high prob)
- "table" (medium prob)
- "banana" (very low)

The model assigns the probability to every word that it knows - typically 50000 to 100000 words. It picks the most likely option and outputs the word. Then it adds the word to the input and predicts again. The cycle continues until the model generates a complete response.

## How it learns the patterns:

The model knows the next word by learning the patterns from the training examples. During the training process, the model processed billions of sentences from books, websites, and articles. It saw many sentences like:

- "The capital of Viet Nam is Ha Noi"
- "Ha Noi is the capital of Viet Nam"

After seeing thousands of examples, the model learnt the statistical relationship between "capital of Viet Nam" and "Ha Noi". When you type the phrase, the model recognizes the pattern and predicts the next word with high confidence.

The model does not have the table of facts. It learnt the statistical pattern from the text, similar to how you notice "once upon a time" usually states the beginning of fairy tales after reading many stories.

## Training vs Inference:

The process of learning from billions of examples is called training. This happens once, before the model is deployed, and requires massive computational resources with thousands of GPUs running for weeks or months.

When you use the model like ChatGPT or Claude, you are using the model in inference mode. The model is not learning anymore - its patterns are frozen. It is applying the patterns that it has already learnt in the training phase to predict the next word. Inference is usually cheaper than training, this is also the reasons why you can get responses in a few seconds.

## What LLM cannot do:

- They do not have real-time information: The model's knowledge freezes at its training cutoff date. It cannot access the current news, stock prices, or your private data unless you provide it.
- They make mistakes confidently: LLM hallucinatesm they generate plausible-sounding but incorrect information. They cannot distinguish between the facts from reliable sources and patters extracted from unrealiable text.
- They do not truly understand: The model predicts text based on statistical patterns. It does not understand meaning like the way humans do. 
- They are not deterministic: The same prompt can produce different outputs. Randomess is built into sampling process that converts probabilities into actual words.