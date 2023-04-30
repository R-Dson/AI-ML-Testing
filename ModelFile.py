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

tf.random.set_seed(7331)

config = {
            "dropout_rate": random.uniform(0.01, 0.80),
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": "accuracy",
            "epochs": 4,
            "batch_size": 64
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
checkpoint = 'Epidot/TwitchLeagueBert-1000k'#'bert-base-uncased'#''#'bert-base-uncased'
newTokenizer = AutoTokenizer.from_pretrained(checkpoint)
newTokenizer.max_length = 512


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
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            WandbMetricsLogger(log_freq=1000),
            WandbModelCheckpoint("models")])

    return modelTrain

def trainModel(data, value, model, tok, Extra=False):
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    
    y = np.array(value, dtype=np.int32)

    #y = np.array([np.array([0, 1]) if yi == 0 else np.array([1, 0]) for yi in y], dtype=np.int32)

    sequences = tok(data, padding='max_length', max_length=512, truncation=True)
    sequences = np.array(sequences['input_ids'])
    #print(type(sequences))
    #print(type(sequences[0]))
    #seq = np.array([])
    """
    l = len(sequences[0])
    print(l)
    for i in sequences:
      if len(i) != l:
        print(len(i))
      np.append(seq, np.array(i))
      #try:
      #  print(np.array(i))
      #except:
      #  print(i)
    sequences = seq#np.array( np.array(x) for x in sequences)#.data['input_ids'])#np.array([np.array(i) for i in sequences.data['input_ids']])
    #print(sequences)#.data['input_ids'])
    print(type(sequences))
    print(type(sequences[0]))"""
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(sequences, y, test_size=1 - train_ratio, random_state=35)

    #X_train_text, X_test_text  = train_test_split(X_train_text, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)

    X_val_text, X_test_text, y_val, y_test= train_test_split(X_test_text, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
    
    print('Loading model...')
    maxLen = len(sequences[0])
    #print(maxLen)
    #print('==================================================')
    #ngram_size = len(cv.vocabulary_)
    vocab_size = tok.vocab_size - 1
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
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            WandbMetricsLogger(log_freq=1000),
            WandbModelCheckpoint("models")])

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
    extraF = files.File(EXTRA_DATA)
    extraF.splitExtra()

    return extraF

def get_data(listBanned, listChat):
    fFUllList = []

    for i in range(len(listBanned)):
        # BANNED
        pathBanned = BANNED_PATH + listBanned[i]
        pathAllowed = CHAT_PATH + listChat[i]
        bannedF = files.File(pathBanned)
        bannedF.bl = bannedF.bl#[:round(len(bannedF.bl))]
        fFUllList.extend([(s.strip() + "\t1" if '\ten\t' in s else '') if ENOnly == True else s.strip() + "\t1" for s in bannedF.bl])

        # ALLOWED
        allowedF = files.File(pathAllowed)
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
        yield allowedF.text


def start(tokenizer, newTokenizerSource):
    print(tf.config.list_physical_devices('GPU'))
    listBanned = os.listdir(BANNED_PATH)
    listChat = os.listdir(CHAT_PATH)
    extra = extraFiles()
    #print(newTokenizer('docLeave', padding='max_length', max_length=512, truncation=True))
    
    #for i in range(len(listBanned)):
      #allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])
      #if newTokenizerSource == False:
        #print(tokenizer.vocab_size)
        #texts = allowedF.text
    filetexts = get_data(listBanned, listChat)
    vocsize = tokenizer.vocab_size
    print('training existing tokenizer')
    print(vocsize)
    tokenizer = tokenizer.train_new_from_iterator(filetexts, show_progress=True, vocab_size=vocsize+5000)
    print(str(tokenizer.vocab_size))
    #print(newTokenizer('docLeave', padding='max_length', max_length=512, truncation=True))
    return extra, listBanned, listChat, tokenizer

def getDataFromPath(pathBanned, pathAllowed):
    fFUllList = []

    # BANNED
    bannedF = files.File(pathBanned)
    bannedF.bl = bannedF.bl#[:round(len(bannedF.bl))]
    fFUllList.extend([(s.strip() + "\t1" if '\ten\t' in s else '') if ENOnly == True else s.strip() + "\t1" for s in bannedF.bl])

    # ALLOWED
    allowedF = files.File(pathAllowed)
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

async def test (tok):
    try:
        model = keras.models.load_model(MODEL_PATH)
        print('continue')
        t = twitchBot('qjrrpzrombru7ghf5nf4pjmlgr6uka', '1x8fy8co8dbhdjk3m8iey4zof57pbt', model, min=30)
        print('continue')
        
        await (t.func(0, 0, tok, 0, None, 0, 0, 0, None, model, MODE_READ, min))
        #asyncio.run(t.func(0, 0, newTokenizer, 0, 0, 0, 0, 0, 0, model, 'READ', min))
    except:
        print('fail')

if __name__ == '__main__':
    existing = []
    try:
        file = open(JSON_PATH, mode="r")
        existing = json.loads(file.read())
        file.close()
    except:
        pass

    wandb.init()
    newTokenizerSource = False
    try:
        with open(TOKEN_PATH, 'rb') as handle:
            newTokenizer = pickle.load(handle)
    except:
        newTokenizerSource = True
        pass

    model = None

    try:
        model = keras.models.load_model(MODEL_PATH)
    except:
        model = None
    train = True
    if train:
        extra, listBanned, listChat, tok = start(newTokenizer, newTokenizerSource)

        if newTokenizerSource:
            with open(TOKEN_PATH, 'wb') as handle:
                pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('train')
        
        if len(listBanned) + len(listChat) != len(existing):
            for i in range(len(listBanned)):
                tempList = listBanned[0:i+1]
                tempList.extend(listChat[0:i+1])

                if listBanned[i] not in existing and listChat[i] not in existing:
                    print(listBanned[i])
                    allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])
                    n = 20000
                    for j in range(n, len(allowedF.text), n):
                        print('train')
                        model = trainModel(allowedF.text[j-n:j], allowedF.value[j-n:j], model, tok)
                        
                    tempList = listBanned[0:i+1]
                    tempList.extend(listChat[0:i+1])
                    tempList = list(set(tempList))
                    saveModel(model, tempList)
    print('done')
    try:
        with open(TOKEN_PATH, 'rb') as handle:
            tok = pickle.load(handle)
        asyncio.run(test(tok))
    except:
        pass