import os
import requests
import tiktoken
import numpy as np
"""
FILE_PATH = "out_data.txt"
def get_data():
    xs = []
    ys = []
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
            #print(xx)
            #print(yy)
               
            xs.append(xx)
            ys.append(yy)
    return xs, ys

data = get_data()[0]
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

# export to bin files
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
"""
# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
"""
import numpy as np
import os

TRAIN_FILE_PATH = "data/moves/train.bin"
VAL_FILE_PATH = "data/moves/val.bin"
FILE_PATH = "out_data.txt"
N = 500000  # Save .bin files every Nth line

def get_data():
    xs = []
    ys = []
    with open(FILE_PATH, "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split(',')
            x = line[0].split('|')
            y = line[1].split('|')
            try:
                xx = [int(i) for i in x]
                yy = [int(i) for i in y]
            except:
                break
            xs.append(xx)
            ys.append(yy)
            
            if (i + 1) % N == 0:
                extend_bin_file(xs, ys)
                xs = []
                ys = []
                print(f"Processed {i + 1:,} lines")
    
    # Save the remaining lines if the total number of lines is not divisible by N
    if len(xs) > 0 and len(ys) > 0:
        extend_bin_file(xs, ys)
    
def extend_bin_file(xs, ys):
    train_data = xs[:int(len(xs) * 0.9)]
    val_data = xs[int(len(xs) * 0.9):]
    
    train_ids = np.array(train_data, dtype=np.uint16)
    val_ids = np.array(val_data, dtype=np.uint16)
    
    # Append train_ids to the existing train.bin file
    with open(TRAIN_FILE_PATH, "ab") as train_file:
        train_ids.tofile(train_file)
    
    # Append val_ids to the existing val.bin file
    with open(VAL_FILE_PATH, "ab") as val_file:
        val_ids.tofile(val_file)
    
    print(f"Extended {TRAIN_FILE_PATH} ({len(train_data):,} tokens)")
    print(f"Extended {VAL_FILE_PATH} ({len(val_data):,} tokens)")

get_data()
"""

import chess.pgn
import numpy as np
import os

pgn_file = open('lichess_db_standard_rated_2014-07.pgn')
FILE_NAME = 'out_data.txt'
TRAIN_FILE_PATH = "data/moves/train.bin"
VAL_FILE_PATH = "data/moves/val.bin"
Y_TRAIN_FILE_PATH = "data/moves/y_train.bin"
Y_VAL_FILE_PATH = "data/moves/y_val.bin"
N = 10000  # Save .bin files every Nth line

data = []

def extend_bin_file(xs, ys):
    train_data = xs[:int(len(xs) * 0.9)]
    y_train = ys[:int(len(ys) * 0.9)]
    val_data = xs[int(len(xs) * 0.9):]
    y_val = ys[int(len(xs) * 0.9):]

    train_ids = np.array(train_data, dtype=np.uint16)
    y_train_ids = np.array(y_train, dtype=np.uint16)
    val_ids = np.array(val_data, dtype=np.uint16)
    y_val_ids = np.array(y_val, dtype=np.uint16)

    # Check if the file exists before appending to it
    if os.path.isfile(TRAIN_FILE_PATH):
        with open(TRAIN_FILE_PATH, "ab") as train_file:
            train_ids.tofile(train_file)
        print(f"Extended {TRAIN_FILE_PATH} ({len(train_data):,} tokens)")
    else:
        with open(TRAIN_FILE_PATH, "wb") as train_file:
            train_ids.tofile(train_file)
        print(f"Created {TRAIN_FILE_PATH} ({len(train_data):,} tokens)")

    if os.path.isfile(VAL_FILE_PATH):
        with open(VAL_FILE_PATH, "ab") as val_file:
            val_ids.tofile(val_file)
        print(f"Extended {VAL_FILE_PATH} ({len(val_data):,} tokens)")
    else:
        with open(VAL_FILE_PATH, "wb") as val_file:
            val_ids.tofile(val_file)
        print(f"Created {VAL_FILE_PATH} ({len(val_data):,} tokens)")

    if os.path.isfile(Y_TRAIN_FILE_PATH):
        with open(Y_TRAIN_FILE_PATH, "ab") as val_file:
            y_train_ids.tofile(val_file)
        print(f"Extended {Y_TRAIN_FILE_PATH} ({len(y):,} tokens)")
    else:
        with open(Y_TRAIN_FILE_PATH, "wb") as val_file:
            y_train_ids.tofile(val_file)
        print(f"Created {Y_TRAIN_FILE_PATH} ({len(val_data):,} tokens)")

    if os.path.isfile(Y_VAL_FILE_PATH):
        with open(Y_VAL_FILE_PATH, "ab") as val_file:
            y_val_ids.tofile(val_file)
        print(f"Extended {Y_VAL_FILE_PATH} ({len(y):,} tokens)")
    else:
        with open(Y_VAL_FILE_PATH, "wb") as val_file:
            y_val_ids.tofile(val_file)
        print(f"Created {Y_VAL_FILE_PATH} ({len(val_data):,} tokens)")


def parse(board):
    string = str(board).replace('\n', ' ').replace(' ', '')
    string = [ord(x) if x != '.' else 0 for x in string]
    return string

i = 0
while True:
    game = chess.pgn.read_game(pgn_file)
    if game is None:
        break

    i += 1
    if i % 100 == 0:
        print(i)

    board = game.board()

    for move in game.mainline_moves():
        x = parse(board)
        board.push(move)
        y = parse(board)
        data.append([x, y])

    if i % N == 0:
        print("Saving data...")
        if len(data) >= N:
            xs = [x[0] for x in data]
            ys = [x[1] for x in data]
            extend_bin_file(xs, ys)
            data = []

pgn_file.close()
print(len(data))

if len(data) > 0:
    print("Saving remaining data...")
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]
    extend_bin_file(xs, ys)

