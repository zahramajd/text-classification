import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gensim
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.python.keras.layers.wrappers import Bidirectional

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K


# parameters
max_words = 1000
max_len = 150
embedding_dimension = 300
top_freq_word_to_use = 40000

word2vec = []
idx2word = {}
word2idx = {}

empty_tag_location = 0
eos_tag_location = 1
unknown_tag_location = 2
word2idx['<empty>'] = empty_tag_location
word2idx['<eos>'] = eos_tag_location
word2idx['<unk>'] = unknown_tag_location
idx2word[empty_tag_location] = '<empty>'
idx2word[eos_tag_location] = '<eos>'
idx2word[unknown_tag_location] = '<unk>'

def read_data(data_path, label_number):
    with open(data_path, "r", encoding="utf-8") as data_file:
        data_file = data_file.read()
        data_file = data_file.replace("\u220c","")

    data = np.array(data_file.split('\n')).reshape(-1,1)
    label = np.empty(data.shape, dtype=int)
    label.fill(label_number)

    d = np.append(data, label, axis=1)

    return d

def read_word_embedding(file_name):
    
    idx = 3
    temp_word2vec_dict = {}
    temp_word2vec_dict['<empty>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
    temp_word2vec_dict['<eos>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
    temp_word2vec_dict['<unk>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
    model = gensim.models.KeyedVectors.load_word2vec_format(file_name, limit = 40000)
    V = model.index2word
    X = np.zeros((top_freq_word_to_use, model.vector_size))
    for index, word in enumerate(V):
        vector = model[word]
        temp_word2vec_dict[idx] = vector
        word2idx[word] = idx
        idx2word[idx] = word
        idx = idx + 1
        if idx % 10000 == 0:
            print ("working on word2vec ... idx ", idx)
            
    return temp_word2vec_dict

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


temp_word2vec_dict = read_word_embedding('data/wiki.fa.vec')
length_vocab = len(temp_word2vec_dict)
shape = (length_vocab, embedding_dimension)
word2vec = np.random.uniform(low=-1, high=1, size=shape)
for i in range(length_vocab):
    if i in temp_word2vec_dict:
        word2vec[i, :] = temp_word2vec_dict[i]

length_vocab, embedding_size = word2vec.shape

model = Sequential()
model.add(Embedding(length_vocab, embedding_size,
                    input_length=max_len,
                    weights=[word2vec], mask_zero=True,
                    name='embedding_layer'))

model.add(Bidirectional(LSTM(64)))
model.add(Dense(256,name='FC1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(sequences_matrix, y_train,
            batch_size=128, 
            epochs=20,
            validation_split=0.15)



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