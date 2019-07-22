import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, \
    Dropout, Dense, Conv1D, MaxPool1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

from hazm import *


def read_data(data_path, label_number):
    with open(data_path, "r", encoding="utf-8") as data_file:
        data_file = data_file.read()
        data_file = data_file.replace("\u220c", "")

    data = np.array(data_file.split('\n')).reshape(-1, 1)

    ## TODO NEED NORMALIZATION
    # normalizer = Normalizer()
    # nText = normalizer.normalize(x)
    ##

    label = np.empty(data.shape, dtype=int)
    label.fill(label_number)

    d = np.append(data, label, axis=1)

    return d


def testClass():
    model = tf.load
    # TODO Having test platform for each new sentence
    pass


def main():
    ## parameters
    ##TODO 1000 is too high for max_len and all tokens are about 17600 why 10000
    # and about the embedding why 100 ?

    max_words = 15000
    max_len = 70
    embedding_size = 150

    num_filters = [64,128,256,512]
    filter_sizes = [3, 4, 5,7]
    drop = 0.5
    batch_size = 30
    epochs = 20

    dictionaryOfTypes = {5: "انتقال اطلاعات",
                         2: "دستوري",
                         6: "روايت كردن مطلب",
                         0: "پرسشي",
                         4: "نقل قول",
                         1: "خواهشي",
                         3: "تهديد"}

    data_a = read_data("data/CleanedSAs/a.txt", label_number=0)
    data_b = read_data("data/CleanedSAs/b.txt", label_number=1)
    data_c = read_data("data/CleanedSAs/c.txt", label_number=2)
    data_d = read_data("data/CleanedSAs/d.txt", label_number=3)
    data_e = read_data("data/CleanedSAs/e.txt", label_number=4)
    data_f = read_data("data/CleanedSAs/f.txt", label_number=5)
    data_g = read_data("data/CleanedSAs/g.txt", label_number=6)


    #### Data Augmentation this also needs new method IF WE WANT Augument or not
    ##TODO data augmentation


    data = np.concatenate((data_a, data_b, data_c, data_d, data_e, data_f, data_g), axis=0)
    np.random.shuffle(data)

    x = data[:, 0]

    y = data[:, 1]
    y = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, shuffle=False)

    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(x_train)
    sequences = tok.texts_to_sequences(x_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

    ## model
    inputs = Input(shape=(max_len,), dtype='int32')
    embedding = Embedding(max_words, embedding_size, input_length=max_len)(inputs)
    reshape = Reshape((max_len, embedding_size, 1))(embedding)

    conv_0 = Conv2D(num_filters[0], kernel_size=(filter_sizes[0], embedding_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters[1], kernel_size=(filter_sizes[0], embedding_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters[2], kernel_size=(filter_sizes[0], embedding_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_3 = Conv2D(num_filters[3], kernel_size=(filter_sizes[0], embedding_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_3)

    # concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2,maxpool_3])
    flatten = Flatten()(maxpool_3)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=7, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights_cnn.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='auto')

    ### TODO check for Deference between checkpoint and model and how to use it


    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Traning Model...")
    history = model.fit(sequences_matrix, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=[checkpoint], validation_split=0.15)

    model.save_weights('weights_cnn.h5')

    test_sequences = tok.texts_to_sequences(x_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    # finding the errors in predicting
    mypredicting = model.predict(test_sequences_matrix)
    mypredicting_max = np.empty_like(mypredicting)
    for i in range(len(mypredicting)):
        mypredicting_max[i, :] = np.int8(mypredicting[i, :] == mypredicting[i, :].max())
    comparing = np.all(mypredicting_max == y_test, axis=1)
    with open("./results/wrongAnswers/" + "cnn-no-embedding" + '.csv', 'w') as f:
        f.write("جمله" + "," + "نوع تشخیص" + "," + "پاسخ درست" + "\n")
        for i in range(len(comparing)):
            if comparing[i] == False:
                f.write(x_test[i])
                f.write("," + dictionaryOfTypes.get(np.argmax(mypredicting_max[i, :])))
                f.write("," + dictionaryOfTypes.get(np.argmax(y_test[i, :])))
                f.write("\n")

    ## MyFunny Testing
    print("اگر دوست دارید متن وترد کنید و اگر نه e را بفرستید")
    a = [input()]
    while a != ["e"]:
        a_test_sequences = tok.texts_to_sequences(a)
        a_test_sequences_matrix = sequence.pad_sequences(a_test_sequences, maxlen=max_len)
        a_mypredicting = model.predict(a_test_sequences_matrix)
        a_mypredicting_max = np.empty_like(a_mypredicting)
        a_mypredicting_max[0, :] = np.int8(a_mypredicting[0, :] == a_mypredicting[0, :].max())
        print(dictionaryOfTypes.get(np.argmax(a_mypredicting_max[0, :])))
        print("اگر دوست دارید متن وترد کنید و اگر نه e را بفرستید")
        a = [input()]
    ####################################################################################

    ###

    ### LOAD and TEST model this needs new method I think
    # TODO this task

    ### k-fold analysis
    #TODO this task

    ### Confutsion Matrix
    ## TODO this task
    # labels = list(y_test)
    # predictions = list(mypredicting)
    # confusion_matrix = tf.confusion_matrix(labels, predictions)
    # print(confusion_matrix)


    #### plot
    plt.style.use('ggplot')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy, Test is: {:0.3f}'.format(accr[1]))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss, Test is: {:0.3f}'.format(accr[0]))
    plt.legend()
    plt.savefig("./results/" + "cnn-no-embedding2")
    plt.show()


if __name__ == '__main__': main()
