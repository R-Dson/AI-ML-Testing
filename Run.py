import pickle
import random
import numpy as np
from twitchBot import *
import Files as files
import json
import transformer as t
import wandb
import asyncio
import tiktoken
import torch
import torch.nn
from torch.nn.utils.rnn import pad_sequence
from numpy import random
import math
import os
from niacin.augment import randaugment
from niacin.text import en
from torch.utils.data import DataLoader
import time
import csv
from torch.nn.functional import pad

enc = tiktoken.get_encoding("cl100k_base")#encoding_for_model("gpt-3.5-turbo")#
assert enc.decode(enc.encode("LLM")) == "LLM"
#tf.random.set_seed(7331)
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/var/home/robin/miniconda3/pkgs/cuda-nvcc-12.1.105-0
config = {
            "dropout_rate": 0.25,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": "accuracy",
            "epochs": 32,
            "batch_size": 64,
            "vocab_size": enc.eot_token,
            "num_heads": 8,
            "embed_dim": 16,
            "num_layers": 8,
            "MAXLEN" : 256
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
EXTRA_CHAT_PATH = 'data/etc/'
EXTRA_DATA = 'data/WikiQA-train.txt'
EXTRA_HATE = 'data/csv/'
JSON_PATH = 'data/usedFiles.json'
TOKEN_PATH = 'data/tokenizer.pickle'
TFIDF_PATH = 'data/TFIDF.pickle'


ENOnly = False
#checkpoint = 'Epidot/TwitchLeagueBert-1000k'#'bert-base-uncased'#''#'bert-base-uncased'
#newTokenizer = AutoTokenizer.from_pretrained(checkpoint)
#newTokenizer.max_length = 512

augmentor = randaugment.RandAugment([en.word.add_synonyms, 
                                     en.word.add_misspelling, 
                                     en.word.swap_words, 
                                     en.word.add_hypernyms,
                                     en.word.add_parens,
                                     en.char.add_fat_thumbs,
                                     en.char.add_characters,
                                     en.char.add_leet], n=3, m=20)
augmentor._p = 0.75

lossfn = None
optimizer = None
#sys.setrecursionlimit(1500)

def trainModel(data, value, model, tok, Extra=False):
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    
    
    print('Adding extra data')
    repeats = 2
    newdata = []
    newys = []
    for i in range(repeats):
        for j in range(len(data)):
            if value[j] == 0:
                if random.randint(1, 10) > 5:
                    continue
                old = data[j]
                for x in augmentor:
                    old = x(old)
                newdata.append(old)
                newys.append(value[j])
            print(str(j) + ' of ' + str(len(data)))

    value.extend(newys)
    data.extend(newdata)
    
    print('Data added')

    newy = []
    for ys in value:
        newy.append([ys])
    value = newy
    y = np.array(value, dtype=np.int16)
    

    print('Preprocessing')
    #y = [([0, 1]) if yi == 0 else ([1, 0]) for yi in y]
    y = torch.tensor(y)
    ans = enc.encode_batch(data)
    z = []
    for x in ans:
        z.append(torch.tensor([l+1 for l in x][:(MAXLEN-len(x))]))
    ans = z
    print('Padding')
    MAXLEN = config['MAXLEN']
    #ans = torch.tensor(ans)
    sequences = [ x for x in ans ]#pad_sequence(ans, batch_first=True)#(ans, padding='post', maxlen=MAXLEN, truncating='post')
    sequences = sequences[:, :MAXLEN]
    
    vocab_size = config['vocab_size']
    
    num_heads = config['num_heads']
    embed_dim = config['embed_dim']
    ff_dim = 4 * embed_dim
    num_layers = config['num_layers']
    dropout = config['dropout_rate']
    
    if model is None:
        print('Generating new model.')
        model = t.Transformer(vocab_size, embed_dim, ff_dim, num_heads, num_layers, dropout, MAXLEN) #t.Transformer(embed_dim, num_layers, num_heads, vocab_size, CUDA=True)

        p = model.count_parameters()
    print('Splitting data')
    validation_ratio = 0.1
    """
    traindata = sequences[:int(len(data)*(1-validation_ratio))]
    trainans = y[:int(len(data)*(1-validation_ratio))]

    valdata = sequences[int(len(data)*(1-validation_ratio)):]
    valans = y[int(len(data)*(1-validation_ratio)):]

    train_size = int(len(data) * train_ratio)"""
    train_size = int(len(data) * train_ratio)
    validation_size = int(len(data) * validation_ratio)
    test_size = int(len(data) * test_ratio)

    traindata = sequences[:train_size]
    trainans = y[:train_size]

    valdata = sequences[train_size:train_size+validation_size]
    valans = y[train_size:train_size+validation_size]

    testdata = sequences[train_size+validation_size:train_size+validation_size+test_size]
    testans = y[train_size+validation_size:train_size+validation_size+test_size]


    if len(traindata) < 1:
        return model

    test_hyperparameter_set = [
    {'lr': 0.001, 'batch_size': 16},
    {'lr': 0.01, 'batch_size': 32},
    {'lr': 0.0001, 'batch_size': 64}
    ]

    for hyperparameters in test_hyperparameter_set:
    # ... Rest of the code ...

    # Train the model with the current hyperparameters
        model.trainModel(traindata, trainans, valdata, valans, testdata, testans, num_epochs=config['epochs'], batch_size=config['batch_size'])

        # Evaluate the model on the validation data
        validation_score = model.evaluate(valdata, valans)

        # Evaluate the model on the test data
        test_score = model.evaluate(testdata, testans)

    model.trainModel(traindata, trainans, valdata, valans, num_epochs=config['epochs'], batch_size=config['batch_size'])


    #l = torch.argmax(valans, dim=1).numpy()
    nz = np.count_nonzero(valans)
    score = model.evaluate(valdata, valans)

    print("Val score: " + str(score))
    print("ratio of zeros: " + str(nz/valans.shape[0]))
    torch.save(model.state_dict(), MODEL_PATH)
    return model

def trainModel2(data, value, model, tok, Extra=False):
    train_ratio = 0.8
    validation_ratio = 0.2
    test_ratio = 0.0

    MAXLEN = config['MAXLEN']
    vocab_size = config['vocab_size']

    num_heads = config['num_heads']
    embed_dim = config['embed_dim']
    ff_dim = 4 * embed_dim
    num_layers = config['num_layers']
    dropout = config['dropout_rate']

    if model is None:
        print('Generating new model.')
        model = t.Transformer(vocab_size, embed_dim, ff_dim, num_heads, num_layers, dropout, MAXLEN)

        p = model.count_parameters()
        print('Number of parameters in model: ' + str(p))

    print('Adding extra data')
    repeats = 0
    newdata = []
    newys = []
    for i in range(repeats):
        for j in range(len(data)):
            if True:#value[j] == 0:
                if random.randint(1, 10) > 3:
                    #continue
                    old = data[j]
                    for x in augmentor:
                        old = x(old)
                    newdata.append(old)
                    newys.append(value[j])
            if j % 10000 == 0:
                print("{:.2f} %".format(j/len(data)*100))

    if len(newdata) > 0:    
        value.extend(newys)
        data.extend(newdata)

    print('Data added. Length: ' + str(len(data)))

    #newy = []
    #for ys in value:
    #    newy.append([ys])
    value = [ [y] for y in value]#newy
    y = np.array(value, dtype=np.int16)

    print('Preprocessing')
    y = torch.tensor(y)
    ans = enc.encode_batch(data)
    #z = []
    #for x in ans:
    #    z.append(torch.tensor([l+1 for l in x]))
    #ans = z
    ans = [ [l+1 for l in x] for x in ans ]
    print('Padding')
    for i in range(len(ans)):
        v = ans[i]
        #if
        #ans[i] = 
        if MAXLEN - len(v) >= 0:
            ans[i].extend([0]*(MAXLEN - len(v))) #pad_sequence(ans, batch_first=True)
            ans[i] = torch.tensor(ans[i])
        else:
            a = torch.tensor(v[:(MAXLEN - len(v))]) #ans[:(MAXLEN - len(ans))] #pad_sequence(ans, batch_first=True)
            ans[i] = a
    for x in ans:
        if len(x.shape) > 1:
            ads = 2
    sequences = ans# torch.tensor(ans)
    #sequences = sequences[:, :MAXLEN]

    print('Splitting data')

    train_size = int(len(data) * train_ratio)
    validation_size = int(len(data) * validation_ratio)
    #test_size = int(len(data) * test_ratio)

    traindata = sequences[:train_size]
    trainans = y[:train_size]

    valdata = sequences[train_size:train_size+validation_size]
    valans = y[train_size:train_size+validation_size]

    #testdata = sequences[train_size+validation_size:train_size+validation_size+test_size]
    #testans = y[train_size+validation_size:train_size+validation_size+test_size]

    bs = config['batch_size']
    bs = int(bs/2)
    
    train_dataloader = DataLoader(list(zip(traindata, trainans)), shuffle=True, batch_size=bs, pin_memory=True, num_workers=4)
    eval_dataloader = DataLoader(list(zip(valdata, valans)), shuffle=True, batch_size=int(bs*4), pin_memory=True, num_workers=4)

    if len(traindata) < 1:
        return model

    # Perform hyperparameter tuning using validation data
    #best_loss = -float('inf')
    best_hyperparameters = None


    lr = 0.0001
    hyperparameter_set = [
        #{'lr': 0.001, 'batch_size': int(bs/4)},
        {'lr': 0.01, 'batch_size': int(bs/2)}
        #{'lr': 0.0001, 'batch_size': bs}
    ]

    #for hyperparameters in hyperparameter_set:
    #    learning_rate = hyperparameters['lr']
    #    batch_size = hyperparameters['batch_size']

        # Create a new model instance with the updated hyperparameters
        #model = t.Transformer(vocab_size, embed_dim, ff_dim, num_heads, num_layers, dropout)

        # Train the model with the current hyperparameters
        #model.trainModel(traindata, trainans, valdata, valans, num_epochs=config['epochs'], batch_size=batch_size, lr=learning_rate)

        # Evaluate the model on the validation data
        #validation_loss = model.evaluate(valdata, valans)

        #if validation_loss > best_loss:
        #    best_loss = validation_loss
        #    best_hyperparameters = hyperparameters

    # Create a new model instance with the best hyperparameters
    best_model = model #t.Transformer(vocab_size, embed_dim, ff_dim, num_heads, num_layers, dropout)
    #best_model.trainModel(traindata, trainans, valdata, valans, num_epochs=config['epochs'], batch_size=best_hyperparameters['batch_size'])
    #model.trainModel(traindata, trainans, valdata, valans, num_epochs=config['epochs'], batch_size=bs, lr=lr)
    nz = 0
    po = 0
    for x in trainans:
            x = list(x.numpy())
            if x == [0]:
                nz += 1
            else:
                po += 1

    print('Starting training')
    #model.TrainTrainer(train_dataloader, eval_dataloader, num_epochs=config['epochs'], batch_size=bs, lr=lr, po=po, nz=nz)
    model.trainModel(train_dataloader, eval_dataloader, num_epochs=config['epochs'], batch_size=bs, lr=lr, po=po, nz=nz)
    #test_loss = best_model.evaluate(testdata, testans)

    print("Best hyperparameters: ", best_hyperparameters)
    #print("Test loss: ", test_loss)
    torch.save(best_model.state_dict(), MODEL_PATH)
    return best_model

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
    #print(tf.config.list_physical_devices('GPU'))
    listBanned = os.listdir(BANNED_PATH)
    listChat = os.listdir(CHAT_PATH)
    extra = extraFiles()
    listExtra = os.listdir(EXTRA_CHAT_PATH)
    listHate = os.listdir(EXTRA_HATE)
    #extraF = files.File(EXTRA_CHAT_PATH)

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
    return extra, listBanned, listChat, tokenizer, listExtra, listHate

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

def getDataFromCSV(filePath, j, offset):
    data = []
    with open(filePath, 'r') as file:
        csvreader = csv.reader(file)
        #rowi = 1
        count = 0
        for row in csvreader:
            count += 1
            #if count == i + offset:
            if count % (j + offset)==0:
                data.append(row)
                #count = 0
            #rowi += 1
    return data

async def test(tok):
    vocab_size = config['vocab_size']
    num_heads = config['num_heads']
    embed_dim = config['embed_dim']
    num_layers = config['num_layers']
    ff_dim = 4 * embed_dim
    MAXLEN = config['MAXLEN']
    model = t.Transformer(vocab_size, embed_dim, ff_dim, num_heads, num_layers, dropout, MAXLEN) 
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    #model = t.TransformerClassifier(vocab_size, embed_dim, num_heads, ff_dim, num_layers)    
    #model.loadModel(MODEL_PATH)
    try:
        #model = keras.models.load_model(MODEL_PATH)
        #model = t.TransformerTranslator()
        
        #model.load_state_dict(torch.load(MODEL_PATH))
        #model.eval()

        print('continue')
        tb = twitchBot('qjrrpzrombru7ghf5nf4pjmlgr6uka', '1x8fy8co8dbhdjk3m8iey4zof57pbt', model, min=30)
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

    #wandb.init()
    newTokenizerSource = False

    model = None

    try:
        vocab_size = config['vocab_size']
        num_heads = config['num_heads']
        embed_dim = config['embed_dim']
        num_layers = config['num_layers']
        maxlen = config['MAXLEN']
        ff_dim = 4 * embed_dim
        dropout = config['dropout_rate']
        model = t.Transformer(vocab_size, embed_dim, ff_dim, num_heads, num_layers, dropout) #t.Transformer(embed_dim, num_layers, num_heads, vocab_size, CUDA=True)
        #model.loadModel(MODEL_PATH)
        try:
            model.load_state_dict(torch.load(MODEL_PATH), strict=False)
        except:
            pass
        p = model.count_parameters()
        print('Model loaded.')
    except:
        model = None
    train = True
    if train:
        extra, listBanned, listChat, tok, listExtra, listHate = start(None, newTokenizerSource)
        tok = enc
        print('train')
        #splitsHate = np.array_split(listHate, len(listBanned))
        splits = np.array_split(listExtra, len(listBanned))

        removedi = []
        
        # https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.p
        #df = pickle.load(open("data/labeled_data.p",'rb'))
        #with open("data/labeled_data.p", 'rb') as handle:
        #    df = pickle.load(handle)
        #    tweets = df.text

        bw = getDataFromCSV('data/refined_ngram_dict.csv', 0, 1)[1:]
        badWords = [x[0] for x in bw]
        offset = 0
        if len(listBanned) + len(listChat) != len(existing):
            for i in range(len(listBanned)-offset):
                i += offset
                s = splits[i]
                torch.cuda.empty_cache()
                allowedF = getDataFromPath(BANNED_PATH + listBanned[i], CHAT_PATH + listChat[i])
                data = allowedF.text
                ans = allowedF.value

                adding = []
                for x in data:
                    if random.randint(0, 11) > 6:
                        xs = x.split(" ")
                        rw = badWords[random.randint(0, len(badWords))]
                        xs[random.randint(0, len(xs))] = rw
                        rw = " ".join(xs)
                        adding.append(rw)

                data.extend(adding)
                ans.extend([BANNED]*(len(adding)))

                for x in splits[i]:
                    f = files.File(EXTRA_CHAT_PATH + x)
                    r = f.splitExtraChat()
                    anss = [0 for _ in range(len(r))]

                    data.extend(r)
                    ans.extend(anss)

                #for h in listHate:
                # 0 = gab.csv
                # 1 = labeled_data.csv
                # 2 = reddit.csv
                firstset = getDataFromCSV(EXTRA_HATE + listHate[0], i, len(listBanned))
                firstset = [" ".join(x[0].split()[1:-1]) for x in firstset]

                temp = getDataFromCSV(EXTRA_HATE + listHate[1], i, len(listBanned))
                firstset.extend([x[6] for x in temp])
                
                temp = getDataFromCSV(EXTRA_HATE + listHate[2], i, len(listBanned))
                for x in temp:
                    if x[1] != 'n/a':
                        firstset.append(" ".join(x[0].split()[1:-1]))
                data.extend(firstset)
                ans.extend([BANNED]*len(firstset))
                #firstset.append([if x[1] == 1 then x[0] else  for x in temp])
                temp = []

                start_time = time.time()
                model = trainModel2(data, ans, model, tok)
                elapsed_time = time.time() - start_time
                print(f'Time: {elapsed_time}. File number ' + str(i+1) + ' of ' + str(len(listBanned)))
                """
                tempList = listBanned
                tempList.extend(listChat)
                tempList = list(set(tempList))
                            
                jl = json.dumps(tempList)
                file = open(JSON_PATH, mode="w")
                file.write(jl)
                file.close()
                print('Wrote json files.')"""

            if False:
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
                            model = trainModel2(data, ans, model, tok)
                            print(str(n+1) + ' of 10. ' + str(i+1) + ' of ' + str(len(listBanned)))

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
    enc = tiktoken.get_encoding("cl100k_base")#encoding_for_model("gpt-3.5-turbo")#get_encoding("cl100k_base")
    asyncio.run(test(enc))
    try:
        #with open(TOKEN_PATH, 'rb') as handle:
            #tok = pickle.load(handle)
        a =2
    except:
        pass
