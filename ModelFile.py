import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
import numpy as np
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.utils import pad_sequences
from keras.models import Sequential, clone_model
from tqdm.keras import TqdmCallback
from keras.layers import Dense, Bidirectional, Embedding, LSTM, BatchNormalization, Activation, Layer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
import asyncio
#import torch
import tiktoken
from niacin.augment import randaugment
#from niacin.text import en
from numpy import random
import math

#import sys

enc = tiktoken.get_encoding("cl100k_base")
#tf.random.set_seed(7331)
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/var/home/robin/miniconda3/pkgs/cuda-nvcc-12.1.105-0
config = {
            "dropout_rate": 0.2,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": "accuracy",
            "epochs": 1,

            "batch_size": 1,
            "vocab_size": enc.eot_token,
            "num_heads": 4,
            "embed_dim": 32,
            "num_layers": 8,
            "MAXLEN" : 64
        }

TEXT = 0
CATEGORY = 1
LANG = 2
NAME = 3
VALUE = 4

ALLOWED = 0
BANNED = 1

MODEL_PATH = './model/model.pt'
CHAT_PATH = 'data/chat/'
BANNED_PATH = 'data/banned/'
EXTRA_DATA = 'data/WikiQA-train.txt'
JSON_PATH = 'data/usedFiles.json'
TOKEN_PATH = 'data/tokenizer.pickle'
TFIDF_PATH = 'data/TFIDF.pickle'

ENOnly = False
#checkpoint = 'Epidot/TwitchLeagueBert-1000k'#'bert-base-uncased'#''#'bert-base-uncased'
#newTokenizer = AutoTokenizer.from_pretrained(checkpoint)
#newTokenizer.max_length = 512

#augmentor = randaugment.RandAugment([en.word.add_synonyms, 
#                                     en.word.add_misspelling, 
#                                     en.word.swap_words, 
#                                     en.word.add_hypernyms], n=3, m=20)
#augmentor._p = 0.75
#enc = tiktoken.encoding_for_model("gpt2") 

lossfn = None
optimizer = None
#sys.setrecursionlimit(1500)

def trainOnData(modelTrain, X_train, y_train, X_val, y_val):
    #print(len(X_train))
    #print(len(y_train))
    #print(len(X_val))
    #print(len(y_val))
    modelTrain.fit(X_train, y_train,
            epochs=config['epochs'],
            verbose=2,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'], 
            shuffle=True,
            callbacks=[TqdmCallback(verbose=2), 
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            WandbMetricsLogger(),
            WandbModelCheckpoint("models")])

    return modelTrain

def trainModel(data, value, model, tok, Extra=False):
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    
    """
    print('Adding extra data')
    repeats = 3
    for i in range(repeats):
        newdata = []
        newys = []
        for i in range(len(data)):
            if random.randint(1, 10) > 9:
                continue
            old = data[i]
            for x in augmentor:
                old = x(old)
            newdata.append(old)
            newys.append(value[i])

        value.extend(newys)
        data.extend(newdata)
    
    print('Data added')

    newy = []
    for ys in value:
        newy.append([ys])
    value = newy"""
    y = np.array(value, dtype=np.int16)

    print('Preprocessing')
    #y = np.array([np.array([0, 1]) if yi == 0 else np.array([1, 0]) for yi in y], dtype=np.int32)
    ans = enc.encode_batch(data)
    z = []
    #for x in ans:
    #    z.append([l+1 for l in x])
    #ans = z
    print('Padding')
    MAXLEN = config['MAXLEN']
    sequences = pad_sequences(ans, padding='post', maxlen=MAXLEN, truncating='post')
    
    #sequences = tok(data, padding='max_length', max_length=512, truncation=True)
    #input_ids = np.array(sequences['input_ids'])
    #attention_mask = np.array(sequences['attention_mask'])
    # create dataset
    #sequences = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, y.astype(np.int32))).batch(config['batch_size'])
    #sequences = np.array(sequences['input_ids'])
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
    MAXLEN = config['MAXLEN']
    
    #X_train_text, X_test_text, y_train, y_test = train_test_split(sequences, y, test_size=1 - train_ratio, random_state=35)

    #X_train_text, X_test_text  = train_test_split(X_train_text, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)

    #X_val_text, X_test_text, y_val, y_test= train_test_split(X_test_text, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
    
    print('Loading model...')
    maxLen = MAXLEN#len(sequences.data['input_ids'][0])
    #print(maxLen)
    #print('==================================================')
    #ngram_size = len(cv.vocabulary_)
    vocab_size = config['vocab_size']
    #vocab_size = ngram_size + vocab_size
    
    num_heads = config['num_heads']
    embed_dim = config['embed_dim']
    ff_dim = 4 * embed_dim
    num_layers = config['num_layers']
    
    if model is None:
        print('Generating new model.')
        
        #model = t.TransformerModelWithMask(maxLen, vocab_size, embed_dim, num_heads, ff_dim, 1, rate=config['dropout_rate'])
        #model = t.TransformerModelWithMask(tok, embed_dim, num_heads, ff_dim, 1, rate=config['dropout_rate'])
        #model = t.TransformerTextModel(tok, 6, d_model=embed_dim, num_heads=num_heads, dff=ff_dim, input_vocab_size=vocab_size, target_vocab_size=1, dropout_rate=0.1)#(tok, embed_dim, num_heads, ff_dim, 1, rate=0.1)

        #model = t.Transformer(maxLen, vocab_size, num_heads, num_layers, embed_dim, ff_dim, 0.1).cuda().to('cuda')
        #model = t.TransformerModel(vocab_size, num_heads, embed_dim, num_layers, ff_dim, MAXLEN)#(vocab_size, embed_dim, num_heads, ff_dim, num_layers, rate=config['dropout_rate']).to('cuda')
        #model.cuda()
        #model = t.TransformerTranslator(embed_dim, num_layers, num_heads, vocab_size, 1, CUDA=True)
        #model = t.TransformerClassifier(vocab_size, embed_dim, num_heads, ff_dim, num_layers, MAXLEN)
        model = t.TransformerEncoderG(num_layers=num_layers, d_model=embed_dim, num_heads=num_heads, dff=ff_dim, input_vocab_size=vocab_size, maximum_position_encoding=MAXLEN, dropout_rate=0.1)
        model.build(input_shape=(None, maxLen))
        model.summary()

        #model = t.TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
        #model = t.LSTMClassifier(embed_dim, ff_dim, vocab_size, num_layers, config['dropout_rate'])
        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        #model.compile_model()
        #optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[config['metrics']])
        #model.build()
        #model.build(input_shape=(None, maxLen))
        #model.summary()
    #wandb.watch(model, log_freq=100, log="all")
    #X_train =  X_train_text#np.concatenate((X_train_text, X_train_freq), axis=1)#[X_train_text, X_train_freq]#X_train_text# [X_train_text,  X_train_freq]#X_train_char, X_train_freq]
    #X_test = X_test_text#np.concatenate((X_test_text, X_test_freq), axis=1)#[X_test_text, X_test_freq]# X_test_text#[X_test_text, X_test_freq] #X_test_char, X_test_freq]
    #train_generator = DataGenerator(data, y, batch_size=config['batch_size'], tokenizer=tok)
    #inp = np.array([ np.array(x) for x in sequences.data['input_ids']] )
    #inp = np.array([ np.array(x) for x in data])
    #input = {'input_ids': np.array(sequences.data['input_ids']), 'attention_mask': np.array(sequences.data['attention_mask'])}


    #input = {'input_ids' : inp, 'attention_mask' : att}
    """
    tokenized_train_data = []
    i = 0
    for example in data:
        text = example
        label = y[i]
        encoded = tok.encode_plus(text, padding='max_length', max_length=512, return_tensors='tf')
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        tokenized_train_data.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label})
        i += 1

    train_dataset = tf.data.Dataset.from_tensor_slices(tokenized_train_data)
    train_dataset = train_dataset.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)"""
    validation_ratio = 0.1
    #data = sequences
    traindata = sequences[:int(len(data)*(1-validation_ratio))]
    trainans = y[:int(len(data)*(1-validation_ratio))]

    valdata = sequences[int(len(data)*(1-validation_ratio)):]
    valans = y[int(len(data)*(1-validation_ratio)):]

    #model.printModel()

    #model.printsummary(config['batch_size'], MAXLEN)
    #train_dataset = torch.utils.data.TensorDataset(traindata, trainans)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    #model.trainModel(traindata, trainans, valdata, valans, epochs=config['epochs'], batch_size=config['batch_size'])
    #e = []
    #for x in enc.encode("test"):
        #e.append(x+1)
    #p = torch.tensor(pad_sequences([e], padding='post', maxlen=MAXLEN, truncating='post'))
    #P = model.pred(p)
    #traindata = torch.tensor(traindata)
    #trainans = torch.tensor(trainans)

    #traindata = tf.convert_to_tensor(traindata, dtype=tf.int32)
    #trainans = tf.convert_to_tensor(trainans, dtype=tf.int32)

    #dataset = tf.data.Dataset.from_tensor_slices((traindata, trainans))
    #shuffle_buffer_size = len(traindata)
    #train_dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).repeat(config['epochs'])
    #train_dataset = train_dataset.batch(config['batch_size'])
    #train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #train_dataset.

    #loss_function = tf.keras.losses.BinaryCrossentropy()
    #optimizer = tf.keras.optimizers.Adam()
    #model.compile(optimizer=optimizer, loss=loss_function)
    if len(traindata) < 1:
        return model
    trainOnData(model, traindata, trainans)
    #model.trainModel()
    #model.fit(train_loader, epochs=config['epochs'], batch_size=config['batch_size'])
    #model.trainer(dataset, loss_function, optimizer, num_epochs=10)
    #model.trainModel(traindata, trainans, config['epochs'], config['batch_size'], lr=1e-3, weight_decay=1e-2)
    #model.trainModel(traindata, trainans, config['epochs'], config['batch_size'], wandb, lr=1e-4, weight_decay=1e-5)
    #model.eval(valdata, valans)

    #loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=batch_size)

    #train_loader = (valdata, valans, batch_size=config['batch_size'], shuffle=True)    
    
    
    #print(str(eval))
    #model = trainOnData(model, traindata, trainans, valdata,)#, X_train_text, y_train, X_test_text, y_test)
    #X_val = X_val_text#np.concatenate((X_val_text, X_val_freq), axis=1)# [X_val_text, X_val_freq]# X_val_text#X_val_char, X_val_freq
    score = model.evaluate(valdata, valans, verbose=True)
    nz = np.count_nonzero(valans)
    #score = model.eval(valdata, valans)

    print("Val score: " + str(score))
    print("ratio of zeros: " + str(nz/valans.size))
    model.save(MODEL_PATH)
    return model

# define a class for the data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, tokenizer):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.indexes = np.arange(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.x[i] for i in batch_indexes]
        batch_y = [self.y[i] for i in batch_indexes]
        batch_encoding = self.tokenizer(batch_x, truncation=True, padding=True, return_tensors="tf")
        return batch_encoding, batch_y

def trainOnData(modelTrain, X_train, y_train, X_val, y_val):
    modelTrain.fit(X_train, y_train,
            epochs=config['epochs'],
            verbose=2,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'], 
            shuffle=True,
            callbacks=[TqdmCallback(verbose=2), 
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            WandbMetricsLogger(),
            WandbModelCheckpoint("models")])

    return modelTrain

def trainOnData(modelTrain, X_train, y):
    modelTrain.fit(X_train, y,
            epochs=config['epochs'],
            verbose=2,
            batch_size=config['batch_size'], 
            shuffle=True,
            callbacks=[TqdmCallback(verbose=2), 
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            WandbMetricsLogger(),
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
    """
    if newTokenizerSource:
        filetexts = get_data(listBanned, listChat)
        vocsize = tokenizer.vocab_size
        print('training existing tokenizer')
        print(vocsize)
        tokenizer = tokenizer.train_new_from_iterator(filetexts, show_progress=True, vocab_size=vocsize + 10000)
        print(str(tokenizer.vocab_size))"""
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

async def test(tok):
    vocab_size = config['vocab_size']
    num_heads = config['num_heads']
    embed_dim = config['embed_dim']
    num_layers = config['num_layers']
    ff_dim = 4 * embed_dim
    MAXLEN = config['MAXLEN']
    #model = t.TransformerClassifier(vocab_size, embed_dim, num_heads, ff_dim, num_layers)    
    #model.loadModel(MODEL_PATH)
    try:
        model = keras.models.load_model(MODEL_PATH)
        #model = t.TransformerTranslator()
        
        #model.load_state_dict(torch.load(MODEL_PATH))
        #model.eval()

        print('continue')
        tb = twitchBot('', '', model, min=30)
        print('continue')
        
        await (tb.func(0, 0, tok, 0, None, 0, 0, 0, None, model, MODE_READ, min))
        #asyncio.run(t.func(0, 0, tok, 0, 0, 0, 0, 0, 0, model, 'READ', min))
    except:
        print('fail')

if __name__ == '__main__':
    existing = []
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
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
        vocab_size = config['vocab_size']
        num_heads = config['num_heads']
        embed_dim = config['embed_dim']
        num_layers = config['num_layers']
        model = t.TransformerTranslator(embed_dim, num_layers, num_heads, vocab_size, CUDA=True)
        model.loadModel(MODEL_PATH)
        #model = torch.load(MODEL_PATH)
        #model = keras.models.load_model(MODEL_PATH)
    except:
        model = None
    train = True
    if train:
        extra, listBanned, listChat, tok = start(None, newTokenizerSource)
        tok = enc
        print('train')

        removedi = []

        if len(listBanned) + len(listChat) != len(existing):
            for n in range(10):
                for i in range(len(listBanned)):
                    if listBanned[i] not in existing and listChat[i] not in existing:
                        allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])
                        dec = n/10
                        inc = (n+1)/10
                        length = len(allowedF.text)
                        data = allowedF.text[round(length * dec):math.floor(length * inc)]
                        ans = allowedF.value[round(length * dec):math.floor(length * inc)]
                        """ones = np.sum(allowedF.value)
                        zers = len(allowedF.value) - ones
                        
                        X = []
                        ys = []
                        i = 0
                        num_z = 0
                        num_o = 0
                        lower = ones if ones < zers else zers
                        lower = int(0.1*lower)
                        while ((num_z != lower) or (num_o != lower)) and i < len(allowedF.value):
                            if i not in removedi:
                                if allowedF.value[i] == 0 and num_z < lower:
                                    X.append(allowedF.text[i])
                                    ys.append(allowedF.value[i])
                                    removedi.append(i)
                                    num_z += 1
                                elif allowedF.value[i] == 1 and num_o < lower:
                                    X.append(allowedF.text[i])
                                    ys.append(allowedF.value[i])
                                    removedi.append(i)
                                    num_o += 1
                                else:
                                    break
                            i += 1
                        X = X[:lower]
                        ys = ys[:lower]"""
                        #model = trainModel(X, ys, model, tok)
                        model = trainModel(data, ans, model, tok)

            tempList = listBanned
            tempList.extend(listChat)
            tempList = list(set(tempList))
                    
            #saveModel(model, tempList)
            jl = json.dumps(tempList)
            file = open(JSON_PATH, mode="w")
            file.write(jl)
            file.close()
            print('Wrote json files.')



        #if newTokenizerSource:
            #w#ith open(TOKEN_PATH, 'wb') as handle:
                #pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """
        if len(listBanned) + len(listChat) != len(existing):
            for i in range(len(listBanned)):
                tempList = listBanned[0:i+1]
                tempList.extend(listChat[0:i+1])

                if listBanned[i] not in existing and listChat[i] not in existing:
                    #print(listBanned[i])
                    allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])
                    #n = 20000
                    #for j in range(n, len(allowedF.text), n):
                    #    print('train')
                    model = trainModel(allowedF.text, allowedF.value, model, tok)
                        #[j-n:j]
                    tempList = listBanned[0:i+1]
                    tempList.extend(listChat[0:i+1])
                    tempList = list(set(tempList))
                    
                    #saveModel(model, tempList)
                    jl = json.dumps(tempList)
                    file = open(JSON_PATH, mode="w")
                    file.write(jl)
                    file.close()
                    print('Wrote json files.')"""
    print('done')

    #checkpoint = 'Epidot/TwitchLeagueBert-1000k'#'bert-base-uncased'
    #newTokenizer = AutoTokenizer.from_pretrained(checkpoint)
    try:
        #with open(TOKEN_PATH, 'rb') as handle:
            #tok = pickle.load(handle)
        #enc = tiktoken.get_encoding("cl100k_base")
        asyncio.run(test(enc))
    except:
        pass