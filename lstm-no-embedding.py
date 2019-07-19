import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt


## parametres
max_words = 1000
max_len = 150
embedding_size = 50
epochs = 30

dictionaryOfTypes = {5: "انتقال اطلاعات",
                        2: "دستوري",
                        6: "روايت كردن مطلب",
                        0: "پرسشي",
                        4: "نقل قول",
                        1: "خواهشي",
                        3: "تهديد"}

def read_data(data_path, label_number):
    with open(data_path, "r", encoding="utf-8") as data_file:
        data_file = data_file.read()
        data_file = data_file.replace("\u220c","")

    data = np.array(data_file.split('\n')).reshape(-1,1)
    label = np.empty(data.shape, dtype=int)
    label.fill(label_number)

    d = np.append(data, label, axis=1)

    return d


data_a = read_data("data/CleanedSAs/a.txt", label_number=0)
data_b = read_data("data/CleanedSAs/b.txt", label_number=1)
data_c = read_data("data/CleanedSAs/c.txt", label_number=2)
data_d = read_data("data/CleanedSAs/d.txt", label_number=3)
data_e = read_data("data/CleanedSAs/e.txt", label_number=4)
data_f = read_data("data/CleanedSAs/f.txt", label_number=5)
data_g = read_data("data/CleanedSAs/g.txt", label_number=6)

data = np.concatenate((data_a, data_b, data_c, data_d, data_e, data_f, data_g), axis=0)
np.random.shuffle(data)

x = data[:,0]

y = data[:,1]
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15)


tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

## model
model = Sequential()
model.add(Embedding(max_words, embedding_size, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(256,name='FC1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(sequences_matrix, y_train,
            batch_size=128, 
            epochs=epochs,
            validation_split=0.15,
            callbacks=[ EarlyStopping(monitor='val_loss', min_delta=0.0001)])

model.save_weights('weights_lstm_no_embedding.h5')


test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# finding the errors in predicting
mypredicting = model.predict(test_sequences_matrix)
mypredicting_max = np.empty_like(mypredicting)
for i in range(len(mypredicting)):
    mypredicting_max[i, :] = np.int8(mypredicting[i, :] == mypredicting[i, :].max())
comparing = np.all(mypredicting_max == y_test, axis=1)
with open("./results/wrongAnswers/" + "lstm-no-embedding" + '.csv', 'w') as f:
    f.write("جمله" + "," + "نوع تشخیص" + "," + "پاسخ درست" + "\n")
    for i in range(len(comparing)):
        if comparing[i] == False:
            f.write(x_test[i])
            f.write("," + dictionaryOfTypes.get(np.argmax(mypredicting_max[i, :])))
            f.write("," + dictionaryOfTypes.get(np.argmax(y_test[i, :])))
            f.write("\n")


#### plot
plt.style.use('ggplot')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'b', label='Training acc')
plt.plot(x, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()