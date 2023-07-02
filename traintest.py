import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
import numpy as np
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
#from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, clone_model
from tqdm.keras import TqdmCallback
from keras.layers import Dense, Bidirectional, Embedding, LSTM, BatchNormalization, Activation, Layer
import os
from matplotlib import pyplot as plt
from twitchBot import *
import Files as files
import json
#import Models as models
from sklearn.model_selection import KFold
import pickle
import transformer as t
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from transformers import AutoTokenizer


#np.random.seed(1337)
tf.random.set_seed(7331)

config = {
            "dropout_rate": random.uniform(0.01, 0.80),
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": "accuracy",
            "epochs": 4,
            "batch_size": 128
        }

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
EXTRA_DATA = 'data/WikiQA-train.txt'
JSON_PATH = 'data/usedFiles.json'
TOKEN_PATH = 'data/tokenizer.pickle'
TFIDF_PATH = 'data/TFIDF.pickle'

ENOnly = False
checkpoint = 'bert-base-uncased'#'Epidot/TwitchLeagueBert-1000k'#'bert-base-uncased'
newTokenizer = AutoTokenizer.from_pretrained(checkpoint)
newTokenizer.max_length = 512
newTokenizer.train()

class File:
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
    
    def split(self, filterList=True):
        for line in self.bl:
            lSplit = line.split('\t')
            try:
                t = lSplit[TEXT]
                if True:#(t not in self.text) or filterList == False:
                    c = int(lSplit[CATEGORY])
                    l = lSplit[LANG]
                    n = lSplit[NAME]
                    v = int(lSplit[VALUE])
                    
                    self.text.append(t)
                    self.cat.append(c)
                    self.lang.append(l)
                    self.name.append(n)
                    self.value.append(v)
            except:
                pass
                #print('failed')

    def splitExtra(self):
        for line in self.bl:
            lSplit = line.split('\t')
            try:
                q = lSplit[0]
                a = lSplit[1]
                
                if True:# q not in self.text:
                    self.text.append(q)
                    self.value.append(ALLOWED)
                if True:# a not in self.text:
                    self.text.append(a)
                    self.value.append(ALLOWED)
            except:
                print('Error loading text.')

def getDataFromPath(pathBanned, pathAllowed):
    fFUllList = []

    # BANNED
    bannedF = File(pathBanned)
    bannedF.bl = bannedF.bl#[:round(len(bannedF.bl))]
    fFUllList.extend([(s.strip() + "\t1" if '\ten\t' in s else '') if ENOnly == True else s.strip() + "\t1" for s in bannedF.bl])

    # ALLOWED
    allowedF = File(pathAllowed)
    allowedF.bl = allowedF.bl#[:round(len(allowedF.bl))]
    for bl in allowedF.bl:
        if ENOnly == True and '\ten\t' not in bl:
            continue
        else:
            fFUllList.extend([bl.strip() + "\t0"])

    fFUllList = list(filter(None, fFUllList))
    random.shuffle(fFUllList)
    allowedF.bl = fFUllList
    allowedF.split(filterList=True)
    return allowedF


def trainOnData(modelTrain, X_train, y_train, X_val, y_val):
    print(len(X_train))
    print(len(y_train))
    print(len(X_val))
    print(len(y_val))
    modelTrain.fit(X_train, y_train,
            epochs=config['epochs'],
            verbose=2,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'], 
            shuffle=True,
            callbacks=[TqdmCallback(verbose=2), 
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])#,
            #WandbMetricsLogger(log_freq=1000),
            #WandbModelCheckpoint("models")])

    return modelTrain

def trainModel(data, value, model, Extra=False):
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    
    y = np.array(value, dtype=np.int32)

    #y = np.array([np.array([0, 1]) if yi == 0 else np.array([1, 0]) for yi in y], dtype=np.int32)

    sequences = newTokenizer(data, padding = 'max_length')
    print(sequences)#.data['input_ids'])
    sequences = np.array(sequences.data['input_ids'])#np.array([np.array(i) for i in sequences.data['input_ids']])

    
    X_train_text, X_test_text, y_train, y_test = train_test_split(sequences, y, test_size=1 - train_ratio, random_state=35)

    #X_train_text, X_test_text  = train_test_split(X_train_text, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)

    X_val_text, X_test_text, y_val, y_test= train_test_split(X_test_text, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
    
    print('Loading model...')
    maxLen = len(sequences[0])
    print(maxLen)
    print('==================================================')
    #ngram_size = len(cv.vocabulary_)
    vocab_size = newTokenizer.vocab_size - 1
    #vocab_size = ngram_size + vocab_size
    
    num_heads = 8
    embed_dim = 256
    ff_dim = 2 * embed_dim
    
    if model is None:
        print('Generating new model.')
        
        model = t.TransformerModel(maxLen, vocab_size, embed_dim, num_heads, ff_dim, 1, rate=config['dropout_rate'])
        model.compile(optimizer=config['optimizer'],
                      loss=config['loss'],
                      metrics=[config['metrics']])
        model.build(input_shape=(None, maxLen))
        model.summary()
    
    #X_train =  X_train_text#np.concatenate((X_train_text, X_train_freq), axis=1)#[X_train_text, X_train_freq]#X_train_text# [X_train_text,  X_train_freq]#X_train_char, X_train_freq]
    #X_test = X_test_text#np.concatenate((X_test_text, X_test_freq), axis=1)#[X_test_text, X_test_freq]# X_test_text#[X_test_text, X_test_freq] #X_test_char, X_test_freq]

    model = trainOnData(model, X_train_text, y_train, X_test_text, y_test)
    #X_val = X_val_text#np.concatenate((X_val_text, X_val_freq), axis=1)# [X_val_text, X_val_freq]# X_val_text#X_val_char, X_val_freq
    score = model.evaluate(X_val_text, y_val, verbose=True)
    print("Val score: " + str(score))
        
    return model

def trainOnData(modelTrain, X_train, y_train, X_val, y_val):
    modelTrain.fit(X_train, y_train,
            epochs=config['epochs'],
            verbose=2,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'], 
            shuffle=True,
            callbacks=[TqdmCallback(verbose=2), 
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])#,
            #WandbMetricsLogger(log_freq=1000),
            #WandbModelCheckpoint("models")])

    return modelTrain

def saveModel(model, files):
    print('Model done. Saving.')
    model.save(filepath=MODEL_PATH)
    print('Saved model.')
    jl = json.dumps(files)
    file = open(JSON_PATH, mode="w")
    file.write(jl)
    file.close()
    print('Wrote json files.')

def extraFiles():
    extraF = File(EXTRA_DATA)
    extraF.splitExtra()

    return extraF

def start():
    print(tf.config.list_physical_devices('GPU'))
    listBanned = os.listdir(BANNED_PATH)
    listChat = os.listdir(CHAT_PATH)
    extra = extraFiles()

    for i in range(len(listBanned)):
      allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])
    
    return extra, listBanned, listChat

existing = []
try:
    file = open(JSON_PATH, mode="r")
    existing = json.loads(file.read())
    file.close()
except:
    pass

longestTokenizer = None
model = None
print('none')
extra, listBanned, listChat = start()
try:
    model = keras.models.load_model(MODEL_PATH)
except:
    model = None
train = True
if train:
    print('train')
    
    if len(listBanned) + len(listChat) != len(existing):
        print('train2')
        for i in range(len(listBanned)):
            print('train3')
            tempList = listBanned[0:i+1]
            tempList.extend(listChat[0:i+1])
            if listBanned[i] not in existing and listChat[i] not in existing:
                print('train4')
                print(listBanned[i])
                allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])
                n = 80000
                for j in range(n, len(allowedF.text), n):
                    print(j)
                    #if j < len(allowedF.text):
                    print('train')
                    model = trainModel(allowedF.text[j-n:j], allowedF.value[j-n:j], model)
                    
                #history.append(historyT)
                #histories.append(historyT.history)
                tempList = listBanned[0:i+1]
                tempList.extend(listChat[0:i+1])
                tempList = list(set(tempList))
                saveModel(model, tempList)
print('done')