import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.types import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, MessageDeletedEvent, ClearChatEvent, LeftEvent,JoinedEvent
import asyncio
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer  
from keras.utils import pad_sequences
from keras.models import Sequential, Model
from tqdm.keras import TqdmCallback
from keras.layers import Dense, Bidirectional, Embedding, LSTM, concatenate, CuDNNLSTM
import os
from matplotlib import pyplot as plt
from attention import AttentionM
from twitchBot import *

TEXT = 0
CATEGORY = 1
LANG = 2
NAME = 3
VALUE = 4

ALLOWED = 0
BANNED = 1

MODEL_PATH = './model'
CHAT_PATH = 'data/chat/'
BANNED_PATH = 'data/banned/'
ExtraData = 'data/WikiQA-train.txt'

ENOnly = False
print(tf.config.list_physical_devices('GPU'))

#print(tf.reduce_sum(tf.random.normal([1000, 1000])))

class Files:
    def __init__(self, fileName):
        b = open(fileName, "r")
        self.bl = b.readlines()
        self.text =  []
        self.cat = []
        self.lang = []
        self.name = []
        self.value = []
        self.question = []
        self.ans = []
    
    def split(self):
        for line in self.bl:
            lSplit = line.split('\t')
            try:
                t = lSplit[TEXT]
                c = lSplit[CATEGORY]
                l = lSplit[LANG]
                n = lSplit[NAME]
                v = lSplit[VALUE]
                
                self.text.append(t)
                self.cat.append(c)
                self.lang.append(l)
                self.name.append(n)
                self.value.append(v)
            except:
                print('failed')

    def splitExtra(self):
        for line in self.bl:
            lSplit = line.split('\t')
            try:
                q = lSplit[0]
                a = lSplit[1]

                self.ans.append(q)
                self.ans.append(a)
                self.question.append(ALLOWED)
                self.question.append(ALLOWED)
            except:
                print('Error loading text.')
        #self.question = list(set(self.question))

extraF = Files(ExtraData)
extraF.splitExtra()

listBanned = os.listdir(BANNED_PATH)
listChat = os.listdir(CHAT_PATH)

fFUllList = []

def TrainOnDataPath(pathBanned, pathAllowed):
    fFUllList = []

    # BANNED
    bannedF = Files(pathBanned)
    bannedF.bl = bannedF.bl[:round(len(bannedF.bl)/10)]
    fFUllList.extend([(s.strip() + "\t1" if '\ten\t' in s else '') if ENOnly == True else s.strip() + "\t1" for s in fBanned.bl])

    # ALLOWED
    allowedF = Files(pathAllowed)
    allowedF.bl = allowedF.bl[:round(len(allowedF.bl)/10)]
    for bl in fChat.bl:
        if ENOnly == True and '\ten\t' not in bl:
            continue
        else:
            fFUllList.extend([bl.strip() + "\t0"])

    fFUllList = list(filter(None, fFUllList))
    random.shuffle(fFUllList)
    allowedF.bl = fFUllList
    allowedF.split()

    print('Loading model...')
    try:
        model = keras.models.load_model(MODEL_PATH)
    except:
        print('Generating new model.')
        model = GenerateAndCompileModel()
    
    trainModel(allowedF, model)

def trainModel(data, model):
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    X = data.text
    y = data.value

    tokenizer = Tokenizer(oov_token="OOV", lower=False, filters='')
    tokenizerChar = Tokenizer(oov_token="OOV", lower=False, char_level=True, filters='')

    tokenizer.fit_on_texts(X)
    tokenizerChar.fit_on_texts(X)

    sequences = tokenizer.texts_to_sequences(X)
    sequencesChar = tokenizerChar.texts_to_sequences(X)

    Xp = pad_sequences(sequences, padding='post')
    maxLen = np.array(Xp).shape[1]

    maxLenPost = np.array(Xp).shape[1]
    num = np.array(Xp).shape[0]#len(X)#

    X = Xp
    y = np.array(y, dtype=np.int64)
    #y = tf.keras.utils.to_categorical(y)
    
    #y = np.array(y, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=0)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0)



    saveModel(model)


def saveModel(model):
    print('Model done. Saving.')
    model.save(filepath=MODEL_PATH)
    print('Saved model.')

for lb in listBanned:
    fBanned = Files(BANNED_PATH + lb)
    fBanned.bl = fBanned.bl[:round(len(fBanned.bl)/10)]
    fFUllList.extend([(s.strip() + "\t1" if '\ten\t' in s else '') if ENOnly == True else s.strip() + "\t1" for s in fBanned.bl])

for lc in listChat:
    fChat = Files(CHAT_PATH + lc)
    fChat.bl = fChat.bl[:round(len(fChat.bl)/10)]
    for bl in fChat.bl:
        if ENOnly == True and '\ten\t' not in bl:
            continue
        else:
            fFUllList.extend([bl.strip() + "\t0"])
            
    #fFUllList.extend([(s.strip() + "\t1" if '\ten\t' in s else '') if ENOnly == True else s.strip() + "\t1" for s in fChat.bl])

"""
fFUllList = list(filter(None, fFUllList))
random.shuffle(fFUllList)
fChat.bl = fFUllList
fChat.split()
"""

fChat.text.extend(extraF.ans)
fChat.value.extend(extraF.question)

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
X = fChat.text
uniqueLang = len(set(fChat.lang))
y = fChat.value

tokenizer = Tokenizer(oov_token="<OOV>", lower=False, filters='')
tokenizerChar = Tokenizer(oov_token="<OOV>", lower=False, char_level=True, filters='')

tokenizer.fit_on_texts(X)
tokenizerChar.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
sequencesChar = tokenizerChar.texts_to_sequences(X)

Xp = pad_sequences(sequences, padding='post')
maxLen = np.array(Xp).shape[1]
mergedseq = []

"""
sclist = []
for j in range(len(sequences)):
    a = Xp[j]
    b = sequencesChar[j]
    
    s = X[j]
    sseq = sequences[j]#tokenizer.texts_to_sequences([s])

    try:
        minil = [0] * max(sseq)
    except:
        minil = [0]
    
    for w in s.split():
        try:
            indx = tokenizer.word_index[w]
            try:
                minil[indx-1] = minil[indx-1] + 1
            except:
                minil[indx-1] = 1
        except:
            pass
    c = np.array(list(a) + b + minil, dtype=np.int32)
    mergedseq.append(c)
    r = j / len(sequences)
    print(r)
Xp = pad_sequences(mergedseq, padding='post')
"""
#Xp = pad_sequences(sequences, padding='post')
"""
tokenizerLang = Tokenizer()
tokenizerLang.fit_on_texts(fChat.lang)
lang_encoded = tokenizerLang.texts_to_sequences(fChat.lang)

hl = []

lang_encoded = [v for lb in lang_encoded for v in lb]
lang_encoded = np.array(lang_encoded)

merged = []
for j in range(np.array(Xp).shape[0]):
    a = Xp[j]
    b = lang_encoded[j]
    merged.append([lang_encoded[j]] + list(Xp[j]))

Xp = np.array(merged)"""
#len(max(sequences, key=len))#
maxLenPost = np.array(Xp).shape[1]
num = np.array(Xp).shape[0]#len(X)#

X = Xp
y = np.array(y, dtype=np.int64)
#y = tf.keras.utils.to_categorical(y)
test = X[0]
#y = np.array(y, dtype=float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=0)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0)

def GenerateModelAndFit():
    model = GenerateModel1()
    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])#['categorical_accuracy'])
    
    history = model.fit(X_train, y_train,
        epochs=4,
        verbose=2,
        validation_data=(X_test, y_test),
        batch_size=128, 
        shuffle=True,
        callbacks=[TqdmCallback(verbose=2), 
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
    
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    return model

def GenerateModel1():
    model = Sequential()
    embedding_dim = 100
    model.add(Embedding(num, embedding_dim, input_length=maxLenPost))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))) 
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.4))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)))
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.6))
    model.add(AttentionM(maxLenPost))
    model.add(keras.layers.Dense(8, activation=tf.nn.relu))
    #model.add(keras.layers.Dropout(0.2))
    model.add(Dense(1,activation=tf.nn.sigmoid))
    return model

def GenerateAndCompileModel():
    model = Sequential()
    embedding_dim = 100
    model.add(Embedding(num, embedding_dim, input_length=maxLenPost))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))) 
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.4))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)))
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.6))
    model.add(AttentionM(maxLenPost))
    model.add(keras.layers.Dense(8, activation=tf.nn.relu))
    #model.add(keras.layers.Dropout(0.2))
    model.add(Dense(1,activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])#['categorical_accuracy'])
    return model

def GenerateModel2():
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model
    
print('Loading model...')
try:
    model = keras.models.load_model(MODEL_PATH)
except:
    print('Generating new model.')
    model = GenerateModelAndFit()
    print('Model done. Saving.')
    model.save(filepath=MODEL_PATH)
    print('Saved model.')

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Val Accuracy: {:.4f}".format(accuracy))

USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
lang = {}
cat = {}

print('Done loading model')

twitchBot('', '', maxLen, maxLenPost, tokenizer, 'tokenizerLang', sequencesChar, model, MODE_READ, min=30)
