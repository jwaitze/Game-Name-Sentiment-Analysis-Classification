# sentiment analysis https://www.kaggle.com/egrinstein/20-years-of-games
# classification
# ign_clean.csv is a modified version of the original. the name is in the first column and the class is in the second

import numpy as np
import tflearn

from tflearn.data_utils import load_csv, pad_sequences, to_categorical
X, y = load_csv('C:/Users/user/Desktop/ign_clean.csv')

classes = list(set(y))
y = [classes.index(y_word) for y_word in y]

def get_char_idx_dict(s):
    d, unique_chars = {}, list(set(s))
    for i, c in enumerate(unique_chars):
        d[c] = i + 1
    return d

max_len, charset = 0, []
for i, x in enumerate(X):
    if len(x[0]) > max_len:
        max_len = len(x[0])
    X[i] = x[0]
    charset = list(set(charset + list(x[0])))

charset_idx = get_char_idx_dict(charset)

def convert_input(s):
    return np.array([charset_idx[c] for c in s] + [0 for i in range(max_len - len(s))])

for i, x in enumerate(X):
    X[i] = np.array([charset_idx[c] for c in x])

X = pad_sequences(X, maxlen=max_len, value=0.)
y = to_categorical(y, len(classes))

net = tflearn.input_data(shape=[None, max_len])
net = tflearn.embedding(net, input_dim=1000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, len(classes), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

model = tflearn.DNN(net)
model.fit(X, y, validation_set=0.1, show_metric=True, batch_size=64)

raw_new_inputs = ['The Big Ginger Beer', 'Killington', 'Snowboard Mania', 'blah', 'ds9a08f']
new_inputs = [convert_input(s) for s in raw_new_inputs]

predictions = model.predict(new_inputs)

def classify_predictions(P):
    classifications = []
    for p in P:
        classifications.append(classes[p.index(max(p))])
    results = {}
    for i, c in enumerate(classifications):
        results[raw_new_inputs[i]] = c
    return results

def prediction_scores(P):
    s = [max(p) for p in P]
    scores = {}
    for i, c in enumerate(s):
        scores[raw_new_inputs[i]] = c
    return scores

results = classify_predictions(predictions)
scores = prediction_scores(predictions)

print(classes)
print(results)
print(scores)
print()

def p(name):
    model.predict([convert_input(s)])[0]
    c = classify_predictions(predictions)
    s = prediction_scores(predictions)
    return (c, s)
