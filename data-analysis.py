from cnn import read_data
import operator
from hazm import *
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob

stemmer = Stemmer()
lemmatizer = Lemmatizer()


def preProcessingAnalysis(data, name):
    charCount = initCharCounter(data)
    wordCount = initWordCounter(data)

    simpleDict, lemmaDict, stemDict = dictionary(data, True, True)

    sorted_charCount = sorted(charCount.items(), key=operator.itemgetter(1))
    sorted_wordCount = sorted(wordCount.items(), key=operator.itemgetter(1))

    sorted_simpleDict = sorted(simpleDict.items(), key=operator.itemgetter(1))
    sorted_lemmaDict = sorted(lemmaDict.items(), key=operator.itemgetter(1))
    sorted_stemDict = sorted(stemDict.items(), key=operator.itemgetter(1))
    listDictSaver2(sorted_charCount, name + "-char-Counter", type='کاراکتر تایی')
    listDictSaver2(sorted_simpleDict, name + "-simpleDict-Counter", type='کلمه')
    listDictSaver2(sorted_lemmaDict, name + "-lemmaDict-Counter", type='کلمه')
    listDictSaver2(sorted_stemDict, name + "-stemDict-Counter", type='کلمه')

    plotting(sorted_charCount, name + "-char-Counter")
    plotting(sorted_wordCount, name + "-word-Counter")


def dictSaver(HDict, name):
    pickle.dump(HDict, open("./results/dictionaries/" + name + ".p", "wb"))


def listDictSaver2(data, name, type):
    with open("./results/data-Analysis/" + name + '.csv', 'w') as f:
        f.write(type + "," + "تعداد" + "\n")
        for key, value in data:
            f.write(str(key))
            f.write("," + "%i" % value)
            f.write("\n")


def listDictSaver(data, name):
    with open(name + '.csv', 'w') as f:
        for value in data:
            f.write(value)


def normalizer(data):
    for x in data:
        normalizer = Normalizer()
        nText = normalizer.normalize(x)
        x = nText
    return data


def initCharCounter(dataFrame):
    counter = 0
    charCount = {}

    for i in dataFrame:
        temp = len(i[0]) - i[0].count(" ")

        if (charCount.get(temp)):
            charCount[temp] = charCount[temp] + 1
        else:
            charCount[temp] = 1

        counter = counter + 1
    return charCount


def initWordCounter(dataFrame):
    counter = 0
    wordCount = {}

    for i in dataFrame:
        temp = len(word_tokenize(i[0]))

        if (wordCount.get(temp)):
            wordCount[temp] = wordCount[temp] + 1
        else:
            wordCount[temp] = 1

        counter = counter + 1
    return wordCount


def dictionary(dataFrame, stemming=False, lemmatize=False):
    stemDict = {}
    dicti = {}
    lemmaDict = {}
    for x in dataFrame:
        k = word_tokenize(x[0])
        for y in k:
            if (dicti.get(y)):
                dicti[y] = dicti[y] + 1
            else:
                dicti[y] = 1

            if stemming:
                stemY = stemmer.stem(y)
                if (stemDict.get(stemY)):
                    stemDict[stemY] = stemDict[stemY] + 1
                else:
                    stemDict[stemY] = 1

            if lemmatize:
                lemmaY = lemmatizer.lemmatize(y)
                if (lemmaDict.get(lemmaY)):
                    lemmaDict[lemmaY] = lemmaDict[lemmaY] + 1
                else:
                    lemmaDict[lemmaY] = 1

    return dicti, lemmaDict, stemDict


def plotting(dataframe, name, color='C0'):
    dataX = [i[0] for i in dataframe]
    dataY = [i[1] for i in dataframe]
    plt.figure()
    plt.bar(dataX, dataY, label=name, color=color)

    plt.ylabel('Frequency')
    plt.xlabel('Tokens')
    plt.title("distribution " + name, y=0.5, loc='right')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("./results/data-Analysis/plot/" + name)


def main():
    fullPath = glob.glob("./data/CleanedSAs/" + "*.txt")

    for i in range(len(fullPath)):
        data = read_data(fullPath[i], label_number=0)
        preProcessingAnalysis(data, str(i))


if __name__ == '__main__': main()
