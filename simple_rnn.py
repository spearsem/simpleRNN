import numpy as np
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.models import Sequential


# Read lines from an example source file.
with open("alice_in_wonderland.txt", 'rb') as _in:
    lines = []
    for line in _in:
        line = line.strip().lower().decode("ascii", "ignore")
        if len(line) == 0:
            continue
        lines.append(line)
text = " ".join(lines)
chars = set([c for c in text])
nb_chars = len(chars)


# Create a character index and reverse mapping to go between a numerical
# ID and a specific character. The numerical ID will correspond to a column
# number when using a one-hot encoded representation of character inputs.
char2index = {c: i for i, c in enumerate(chars)}
index2char = {i: c for i, c in enumerate(chars)}

# For convenience, choose a fixed sequence length of 10 characters.
SEQLEN, STEP = 10, 1
input_chars, label_chars = [], []

# Convert the data into a series of different SEQLEN-length subsequences.
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i: i + SEQLEN])
    label_chars.append(text[i + SEQLEN])


# Compute the one-hot encoding of the input sequences X and the next
# character (the label) y
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1


# Set up a bunch of metaparameters for the network and training regime.
BATCH_SIZE, HIDDEN_SIZE = 128, 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100


# Create a super simple recurrent neural network. There is one recurrent
# layer that produces an embedding of size HIDDEN_SIZE from the one-hot
# encoded input layer. This is followed by a Dense fully-connected layer
# across the set of possible next characters, which is converted to a
# probability score via a standard softmax activation with a multi-class
# cross-entropy loss function linking the prediction to the one-hot
# encoding character label.
model = Sequential()
model.add(
    GRU(  # Note you can vary this with LSTM or SimpleRNN to try alternatives.
        HIDDEN_SIZE,
        return_sequences=False,
        input_shape=(SEQLEN, nb_chars),
        unroll=True
    )
)
model.add(Dense(nb_chars))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


# Execute a series of training and demonstration iterations.
for iteration in range(NUM_ITERATIONS):

    # For each iteration, run the model fitting procedure for a number of epochs.
    print("=" * 50)
    print("Iteration #: %d" % (iteration))
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    # Select a random example input sequence.
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]

    # For a number of prediction steps using the current version of the trained
    # model, construct a one-hot encoding of the test input and append a prediction.
    print("Generating from seed: %s" % (test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH):

        # Here is the one-hot encoding.
        X_test = np.zeros((1, SEQLEN, nb_chars))
        for j, ch in enumerate(test_chars):
            X_test[0, j, char2index[ch]] = 1

        # Make a prediction with the current model.
        pred = model.predict(X_test, verbose=0)[0]
        y_pred = index2char[np.argmax(pred)]

        # Print the prediction appended to the test example.
        print(y_pred, end="")

        # Increment the test example to contain the prediction as if it
        # were the correct next letter.
        test_chars = test_chars[1:] + y_pred
print()
