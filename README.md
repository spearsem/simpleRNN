# simpleRNN
Example RNN for text generation from "Deep Learning With Keras" by Gulli and Pal (Chapter 6).

- This repo is meant to be an ad hoc exploratory script for training a character-generating
  recurrent neural network using an example text from Project Gutenberg. It's not meant to
  provide state of the art performance or any insight into nuanced optimization of this
  sort of model for a real production use case. Just an overview of the basic idea.

- Use the provided `environment.yml` file to create your own complete Python 3.6 environment
  with `conda`, including the specific dependencies needed for this example. E.g.
  `conda env create -f environment.yml` and `source activate py36-keras-simpleRNN`.

- Read through some comments in the code for ways to experiment with changing the network and
  reference the source materials in the book "Deep Learning With Keras" by Gulli and Pal 
  (Chapter 6).

Here is an example of the output you will see by executing the script `simple_rnn.py`:

```
Iteration #: 5
Epoch 1/1
158773/158773 [==============================] - 85s - loss: 1.5204      
Generating from seed:  so she we
 so she went on the dormouse she said to her had not the more to her had not the more to her had not the more
==================================================
Iteration #: 6
Epoch 1/1
158773/158773 [==============================] - 88s - loss: 1.4686      
Generating from seed:  and the d
 and the dormouse she was a little she was a little she was a little she was a little she was a little she was
==================================================
Iteration #: 7
Epoch 1/1
158773/158773 [==============================] - 92s - loss: 1.4250     
Generating from seed:  at the wi
 at the window, and the mock turtle said to herself in a long and said the caterpillar said the caterpillar sa
==================================================
```

So for example, after 7 rounds of training, the network would take the input " at the wi" 
and autocomplete it to the sequence, " at the window, and the mock turtle said to herself
in a long and said the caterpillar said the caterpillar sa ".

Given that the network has only a single hidden recurrent layer and uses only sequences of
10 characters at a time for training, it's impressive the way it learns to mimic the type
of whimsical sentence you might find in "Alice in Wonderland." Of course, for real use cases
there would likely be many more hidden layers with more complicated network topologies, and
there would be effort spent optimizing metaparameters like optimizer choices, learning rates,
hidden layer sizes, and so forth. (Note also, that this is trained only on a CPU laptop, so
the parameters are kept artificially constrained to facilitate faster training).
