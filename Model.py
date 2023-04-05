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
from matplotlib import pyplot as plt
from twitchBot import *
import Files as files
import json
import Models as models
from sklearn.model_selection import KFold
import pickle
import transformer as t


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

def trainModel(data, value, tokenizer, maxLen, num, tokenizerChar, maxLenChar, numChar, cvNum, cv, model=None, Extra=False):
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    #print(len(data))
    X = data#[:10000]
    y = np.array(value, dtype=np.int32)

    #y = np.array([np.array([0, 1]) if yi == 0 else np.array([1, 0]) for yi in y], dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(X)
    
    sequences = pad_sequences(sequences, padding='post', maxlen=tokenizer.num_words)
    #sequencesChar = tokenizerChar.texts_to_sequences(X)
    #sequencesChar = pad_sequences(sequencesChar, padding='post', maxlen=maxLenChar)
    sequencesFreq = cv.transform(X).toarray()
    
    X_train_text, X_test_text = train_test_split(sequences, test_size=1 - train_ratio, random_state=35)
    #X_train_char, X_test_char = train_test_split(sequencesChar, test_size=1 - train_ratio, random_state=35)
    #X_train_freq, X_test_freq, y_train, y_test = train_test_split(sequencesFreq, y, test_size=1 - train_ratio, random_state=35)
    
    #inputs = [X_train_text, X_train_char, X_train_freq]
    #X_test = [X_val_text, X_val_char, X_val_freq]
    #X_test = [X_test_text, X_test_freq]# X_test_char, X_test_freq]
    #X_val = [X_val_text, X_val_char, X_val_freq]
    
    # n = X_train_text.shape[0]
    #nval = X_val_text.shape[0]
    ID_Inp = 1#np.array(range(n))
    ID_Out = 1#np.array(range(n))
    fold_no = 1
    num_folds = 1
    bestHistory = None
    bestModel = None
    bestScore = np.inf# [np.inf, -np.inf]

    if num_folds > 1:
        kfold = KFold(n_splits=num_folds, shuffle=False)
        for train, test in kfold.split(ID_Inp, ID_Out):
            print('Loading model...')
            if model == None:
                print('Generating new model.')
                tmpModel = models.model3(num, numChar, cvNum, maxLen, maxLenChar, len(cv.vocabulary_))
                tmpModel.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            else:
                tmpModel = clone_model(model)
                tmpModel.build((None, maxLen+maxLenChar+len(cv.vocabulary_)))
                tmpModel.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])#['categorical_accuracy'])
                tmpModel.set_weights(model.get_weights())
            print('Training.')
            #Fold_Train_Input1, Fold_Train_Input2, Fold_Train_Input3 = X_train_text[train], X_train_char[train], X_train_freq[train]
            Fold_Train_Input1, Fold_Train_Input3 = 1#X_train_text[train], X_train_freq[train]
            Fold_Train_OutPut = y_train[train]

            tInput = [Fold_Train_Input1, Fold_Train_Input3]

            #Fold_Test_Input1, Fold_Test_Input2, Fold_Test_Input3 = X_train_text[test], X_train_char[test], X_train_freq[test]
            Fold_Test_Input1, Fold_Test_Input3 = 1#X_train_text[test], X_train_freq[test]

            Fold_Test_OutPut = y_train[test] #= X_val_text[test], X_val_char[test], X_val_freq[test]
            

            tTest = [Fold_Test_Input1, Fold_Test_Input3]

            tmpModel, history = trainOnData(tmpModel, tInput, Fold_Train_OutPut, tTest, Fold_Test_OutPut)

            score = tmpModel.evaluate(X_test, y_test, verbose=False)
            #if score[1]-score[0] > bestScore[1]-bestScore[0]:
            
            if bestScore > score[0]:
                bestScore = score[0]
                print("Best val score: " + str(bestScore))
                bestModel = tmpModel
                bestHistory = history
        
        model = bestModel
        history = bestHistory

    else:
        X_train_text, X_test_text = train_test_split(sequences, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
        #X_train_char, X_test_char = train_test_split(sequencesChar, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
        X_train_freq, X_test_freq, y_train, y_test = train_test_split(sequencesFreq, y, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)

        X_val_text, X_test_text = train_test_split(X_test_text, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
        #X_val_char, X_test_char= train_test_split(X_test_char, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
        X_val_freq, X_test_freq, y_val, y_test = train_test_split(X_test_freq, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=35)
        
        print('Loading model...')
        maxlen = tokenizer.num_words
        vocab_size = tokenizer.num_words
        
        num_heads = 2
        embed_dim = 8
        ff_dim = 2 * embed_dim

        if model == None:
            print('Generating new model.')
            #model = models.improved_model(tokenizer.num_words, numChar, cvNum, maxLen, maxLenChar, len(cv.vocabulary_))
            longestlen = len(max(tokenizer.word_docs, key=len))
            #model = t.TransformerModel(longestlen, tokenizer.num_words, num_transformer_blocks, d_model, num_heads, dff)
            model = t.TransformerModel(maxLen, vocab_size, embed_dim, num_heads, ff_dim, 1)


            #model.compile(loss='categorical_crossentropy',
            #optimizer='adam',
            #metrics=['accuracy'])
            #sgd = keras.optimizers.SGD(clipvalue=0.5)
            model.compile(optimizer='adamax',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.build(input_shape=(2, vocab_size))
            #model.build(input_shape=((None, maxlen), (None, vocab_size)))
            model.summary()

        
        X_train = X_train_text# [X_train_text, X_train_freq]# [X_train_text,  X_train_freq]#X_train_char, X_train_freq]
        X_test = X_test_text#[X_test_text, X_test_freq]# [X_test_text, X_test_freq] #X_test_char, X_test_freq]
        
        model, history = trainOnData(model, X_train, y_train, X_test, y_test)
        #odel, history = trainOnDataTransformer(model, X_train_text, X_train_freq, y_train, X_val_text, X_val_freq, y_val) #(model, X_train, y_train, X_test, y_test)

        X_val = X_val_text#[X_val_text, X_val_freq]#X_val_char, X_val_freq]

        score = model.evaluate(X_val, y_val, verbose=False)
        print("Val score: " + str(score))
        
    return model, history

def trainOnData(modelTrain, X_train, y_train, X_val, y_val):
    history = modelTrain.fit(X_train, y_train,
            epochs=4,
            verbose=2,
            validation_data=(X_val, y_val),
            batch_size=2, 
            shuffle=True,
            callbacks=[TqdmCallback(verbose=2), 
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

    return modelTrain, history

def saveModel(model, files):
    print('Model done. Saving.')
    model.save(filepath=MODEL_PATH)
    print('Saved model.')
    jl = json.dumps(files)
    file = open(JSON_PATH, mode="w")
    file.write(jl)
    file.close()
    print('Wrote json files.')

def extraFiles(tokenizer):
    extraF = files.File(EXTRA_DATA)
    extraF.splitExtra()
    maxLen = 0
    longest = 0
    """
    tokenizer = None #.fit_on_texts(extraF.text)
    sequences = tokenizer.texts_to_sequences(extraF.text)
    longest = max(sequences, key=len)

    
    
    if len(longest) > maxLen:
        maxLen = len(longest)
    num = len(sequences)
    #num = 11110
    
    tokenizerChar = Tokenizer(oov_token="OOV", lower=False, char_level=True, filters='')
    
    tokenizerChar.fit_on_texts(extraF.text)
    sequencesChar = tokenizerChar.texts_to_sequences(extraF.text)
    longestChar = max(sequencesChar, key=len)
    
    maxLenChar = 0
    if len(longestChar) > maxLenChar:
        maxLenChar = len(longestChar)
    numChar = len(longestChar)
    
    cv = TfidfVectorizer()#CountVectorizer()#ngram_range=(1,2))
    cv.fit(extraF.text)
    cvNum = len(cv.vocabulary_)
    """
    cvNum = 0#111111110
    cv = None
    tokenizerChar = None
    maxLenChar = 0
    numChar = 0
    num = 0
    return extraF, maxLen, num, tokenizer, tokenizerChar, maxLenChar, numChar, cvNum, cv

def start():
    print(tf.config.list_physical_devices('GPU'))
    listBanned = os.listdir(BANNED_PATH)
    listChat = os.listdir(CHAT_PATH)

    tokenizer = Tokenizer(oov_token="<OOV>", filters='', num_words=2500)
    extra, maxLen, num, longestTokenizer, tokenizerCharLongest, maxLenChar, numChar, cvNum, cv = extraFiles(tokenizer)
    maxLen = 0
    longest = 0
    for i in range(len(listBanned)):
        allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])

        tokenizer = Tokenizer(oov_token="<OOV>", filters='', num_words=2500)
        
        tokenizer.fit_on_texts(allowedF.text)
        sequences = tokenizer.texts_to_sequences(allowedF.text)
        #longest = max(sequences, key=len)
        if tokenizer.document_count > longest:
            longest = tokenizer.document_count
            longestTokenizer = tokenizer
        if len(sequences) > num:
            num = len(sequences)

        tempmaxlen = len(max([x.split() for x in allowedF.text], key=len))
        #print(str(tempmaxlen) + ' vs ' + str(maxLen))
        if tempmaxlen > maxLen:
            maxLen = tempmaxlen
        
        """
        tokenizerChar = Tokenizer(oov_token="OOV", lower=True, char_level=True, filters='', )
        tokenizerChar.fit_on_texts(allowedF.text)
        sequencesChar = tokenizerChar.texts_to_sequences(allowedF.text)
        longestChar = max(sequencesChar, key=len)

        if len(longestChar) > maxLenChar:
            maxLenChar = len(longestChar)
            tokenizerCharLongest = tokenizerChar
        if len(sequencesChar) > numChar:
            numChar = len(longestChar)"""

        cvT = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
        cvT.fit(allowedF.text)
        inp = cvT.transform(allowedF.text).toarray()
        cvNumT = len(cvT.vocabulary_)#arr.shape[1]
        maxlen = 0
        if cvNumT > cvNum:
            cvNum = inp.shape[1]
            cv = cvT
        
    return extra, maxLen, num, None, longestTokenizer, listBanned, listChat, tokenizerCharLongest, maxLenChar, numChar, cvNum, cv

def plot(histories):
    ha = []
    hl = []
    hvl = []
    hva = []
    for i in histories:
        ha.extend(i['binary_accuracy'])
        hl.extend(i['loss'])
        hvl.extend(i['val_loss'])
        hva.extend(i['val_accuracy'])

    plt.plot(ha)
    plt.plot(hva)

    plt.title('model val_accuracy')
    plt.ylabel('val_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(hvl)
    plt.plot(hl)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    np.random.seed(1337)
    tf.random.set_seed(7331)
    
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)

    tf.compat.v1.keras.backend.set_session(sess)
    
    train = True

    existing = []
    try:
        file = open(JSON_PATH, mode="r")
        existing = json.loads(file.read())
        file.close()
    except:
        pass
    
    newModel = False
    longestTokenizer = None
    oldTokenizer = None
    cv = None
    oldcv = None

    try:
        with open(TOKEN_PATH, 'rb') as handle:
            oldTokenizer = pickle.load(handle)
    except:
        pass

    try:
        with open(TFIDF_PATH, 'rb') as handle:
            oldcv = pickle.load(handle)
    except:
        pass
    extra, maxLen, num, longest, longestTokenizer, listBanned, listChat, tokenizerCharLongest, maxLenChar, numChar, cvNum, cv = start()
    try:
        model = keras.models.load_model(MODEL_PATH)
    except:
        model = None
        newModel = True
        et = list(np.array_split(np.array(extra.text, dtype=object), len(listBanned)))

    if train:
        
        history = []
        histories = []

        if len(listBanned) + len(listChat) != len(existing):
            with open(TOKEN_PATH, 'wb') as handle:
                pickle.dump(longestTokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(TFIDF_PATH, 'wb') as handle:
                pickle.dump(cv, handle, protocol=pickle.HIGHEST_PROTOCOL)

            for i in range(len(listBanned)):#range(2):#
                tempList = listBanned[0:i+1]
                tempList.extend(listChat[0:i+1])

                if listBanned[i] not in existing and listChat[i] not in existing:
                    allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])

                    if newModel:
                        allowedF.text.extend(list(et[i]))
                        allowedF.value.extend([ALLOWED] * len(et[i]))

                    n = 20000
                    for i in range(n, len(allowedF.text), n):
                        if oldcv == None and oldTokenizer == None:
                            model, historyT = trainModel(allowedF.text[i-n:i], allowedF.value[i-n:i], longestTokenizer, maxLen, num, None, maxLenChar, numChar, cvNum, cv, model)
                        else:
                            model, historyT = trainModel(allowedF.text[i-n:i], allowedF.value[i-n:i], oldTokenizer, maxLen, num, None, maxLenChar, numChar, cvNum, oldcv, model)
                        
                        history.append(historyT)
                        histories.append(historyT.history)

                    tempList = listBanned[0:i+1]
                    tempList.extend(listChat[0:i+1])

                    tempList = list(set(tempList))
                    saveModel(model, tempList)
            try:
                plot(histories)
            except: 
                pass
    
    if longestTokenizer == None:
        with open(TOKEN_PATH, 'rb') as handle:
            longestTokenizer = pickle.load(handle)

    if cv == None:
        with open(TFIDF_PATH, 'rb') as handle:
            cv = pickle.load(handle)
    
    ID = ''
    SECRET = ''

    twitchBot(ID, SECRET, maxLen, longest, longestTokenizer, None, None, tokenizerCharLongest, maxLenChar, numChar, cv, model, min=30)
