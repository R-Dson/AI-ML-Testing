import chess.pgn
"""
# Load the PGN file
pgn_file = open('lichess_db_standard_rated_2014-07.pgn')

FILE_NAME = 'out_data.txt'

data = []

def parse(board):
    string = str(board).replace('\n', ' ').replace(' ', '')
    string = [ ord(x) if x != '.' else 0 for x in string ]
    return string

i = 0
while True:
    game = chess.pgn.read_game(pgn_file)
    if game is None:
        break

    i+=1
    if i % 100 == 0:
        print(i)

    board = game.board()

    for move in game.mainline_moves():
        x = parse(board)
        board.push(move)
        y = parse(board)
        data.append([x, y])

pgn_file.close()
print(len(data))
f = open(FILE_NAME, "w")
for x in data:
    for i in range(len(x[0])):
        out = x[0][i]
        if i != len(x[0])-1:
            f.write(str(out)+'|')
        else:
            f.write(str(out))
    f.write(',')
    for i in range(len(x[1])):
        out = x[1][i]
        if i != len(x[1])-1:
            f.write(str(out)+'|')
        else:
            f.write(str(out))
    
    f.write('\n')
f.close()"""
import chess.pgn

# Load the PGN file
pgn_file = open('lichess_db_standard_rated_2014-07.pgn')

FILE_NAME = 'out_data.txt'

data = []

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

    if i % 10000 == 0:
        print("Saving data...")
        with open(FILE_NAME, "a") as f:
            for x in data:
                for i in range(len(x[0])):
                    out = x[0][i]
                    if i != len(x[0]) - 1:
                        f.write(str(out) + '|')
                    else:
                        f.write(str(out))
                f.write(',')
                for i in range(len(x[1])):
                    out = x[1][i]
                    if i != len(x[1]) - 1:
                        f.write(str(out) + '|')
                    else:
                        f.write(str(out))

                f.write('\n')
        data = []

pgn_file.close()
print(len(data))

if len(data) > 0:
    print("Saving remaining data...")
    with open(FILE_NAME, "a") as f:
        for x in data:
            for i in range(len(x[0])):
                out = x[0][i]
                if i != len(x[0]) - 1:
                    f.write(str(out) + '|')
                else:
                    f.write(str(out))
            f.write(',')
            for i in range(len(x[1])):
                out = x[1][i]
                if i != len(x[1]) - 1:
                    f.write(str(out) + '|')
                else:
                    f.write(str(out))

            f.write('\n')
