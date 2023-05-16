import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
#print(decode_review(train_data[0]))
max_sequence_len = 256
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=max_sequence_len)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=max_sequence_len)
#for i in range(len(train_data)):
 #   print(len(train_data[i]))
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, 160))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(160, activation='relu'))
# model.add(keras.layers.Dense(80, activation='relu'))
# model.add(keras.layers.Dense(10, activation='relu'))
# model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.summary()

model.summary()
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
partial_x_train = partial_x_train.astype(np.int64)
partial_y_train = partial_y_train.astype(np.int64)
x_val = x_val.astype(np.int64)
y_val = y_val.astype(np.int64)
#print(type(partial_x_train))
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=256,
                    validation_data=(x_val, y_val),
                    verbose=1)
test_data = test_data.astype(np.int64)
test_labels = test_labels.astype(np.int64)
results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)
import matplotlib.pyplot as plt
history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, color = "orchid",linestyle='dotted', label='Training loss')
plt.plot(epochs, val_loss, color = "g",linestyle='--', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
#预测函数检验
#print(decode_review(test_data[0]))
#print(test_labels[0])
#print(decode_review(test_data[1]))
#print(test_labels[1])
#predictions = model.predict(test_data)
#print(predictions)
