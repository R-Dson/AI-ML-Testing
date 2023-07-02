import pickle
import random
import numpy as np
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
from model import GPT, GPTConfig
import train as tr
import param

FILE_PATH = "out_data.txt"
vocab_size = 6*2+1
MAX_LEN = 64

config = {
    "dropout_rate": 0.25,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": "accuracy",
    "epochs": 32,
    "batch_size": 2,
    "vocab_size": vocab_size,
    "num_heads": 1,
    "embed_dim": 4,
    "num_layers": 1,
    "MAXLEN" : MAX_LEN
}

c = GPTConfig()

def trainf(x, y, model=None):
    train_ratio = 0.8
    validation_ratio = 0.2
    test_ratio = 0.0      

    maxlen = config['MAXLEN']
    vocab_size = config['vocab_size']

    num_heads = config['num_heads']
    embed_dim = config['embed_dim']
    ff_dim = 4 * embed_dim
    num_layers = config['num_layers']
    dropout = config['dropout_rate']



    if model is None:
        
        model = GPT(c)

        print('Generating new model.')
        #model = t.Transformer(vocab_size, 1, ff_dim, num_heads, num_layers, dropout, maxlen)
        #model = t.Transformer(vocab_size=vocab_size, seq_len=MAX_LEN, d_model=embed_dim, d_ff=ff_dim, h=num_heads, num_layers=num_layers, dropout=dropout)

        #p = model._parameters.
        #print('Number of parameters in model: ' + str(p))


    p = param.Param()
    t = tr.Train(p)


    train_size = int(len(x) * train_ratio)
    validation_size = int(len(x) * validation_ratio)
    #test_size = int(len(data) * test_ratio)

    traindata = (x[:train_size])
    trainans = (y[:train_size])

    valdata = (x[train_size:train_size+validation_size])
    valans = (y[train_size:train_size+validation_size])

    #testdata = sequences[train_size+validation_size:train_size+validation_size+test_size]
    #testans = y[train_size+validation_size:train_size+validation_size+test_size]

    bs = config['batch_size']
    bs = int(bs/2)
    
    train_dataloader = DataLoader(list(zip(traindata, trainans)), shuffle=True, batch_size=bs)
    eval_dataloader = DataLoader(list(zip(valdata, valans)), shuffle=True, batch_size=int(bs*4))

    lr = 0.0001

    model.trainModel(train_dataloader, eval_dataloader, num_epochs=config['epochs'], batch_size=bs, lr=lr)


    #l = torch.argmax(valans, dim=1).numpy()
    nz = np.count_nonzero(valans)
    avg_loss, score, precision, recall, f1 = model.evaluate3(valdata, valans)

    print("Val score: " + str(score))
    print("ratio of zeros: " + str(nz/valans.shape[0]))
    torch.save(model.state_dict(), 'model.pt')
    return model


def get_data():
    xs = []
    ys = []
    return xs, ys
    with open(FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            x = line[0].split('|')
            y = line[1].split('|')
            #xx = [torch.tensor(int(i), dtype=torch.int16) for i in x]
            #yy = [torch.tensor(int(i), dtype=torch.int16) for i in y]
            xx = [int(i) for i in x]
            yy = [int(i) for i in y]
               
            xs.append(xx)
            ys.append(yy)
    return xs, ys

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    train_model = True

    if train_model:
        x, y = get_data()
        trainf(x, y)

