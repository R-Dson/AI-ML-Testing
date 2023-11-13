import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import string
import chess
import chess.engine
import time
from stockfish import Stockfish
import multiprocessing
import random
from torch.utils.data import DataLoader

from torch.nn import TransformerEncoder, TransformerEncoderLayer


stockfish = Stockfish(
    "stockfish-path"
)
valid_move_tokens = [
    f"{c1}{r}{c2}{r2}"
    for c1 in string.ascii_lowercase[:8]
    for r in string.digits[1:9]
    for c2 in string.ascii_lowercase[:8]
    for r2 in string.digits[1:9]
]

uci_to_int = {uci_move: i for i, uci_move in enumerate(valid_move_tokens)}
custom_vocab_in = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
    "w": 13,
    "b": 14,
    "-": 15,
    "0": 16,
    "1": 17,
    "2": 18,
    "3": 19,
    "4": 20,
    "5": 21,
    "6": 22,
    "7": 23,
    "8": 24,
    "9": 25,
    "e": 26,
    "c": 27,
    "f": 28,
    "%": 29,
    "/": 30,
    " ": 31,
    "#": 32,
    "a": 33,
    "d": 34,
    "g": 35,
    "h": 36,
}

max_length = 128


class ModelConfig:
    def __init__(self, emb_dim=64, n_layers=8, n_heads=8):
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads


class ChessTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, config.emb_dim).cuda()
        encoder_layer = TransformerEncoderLayer(config.emb_dim, config.n_heads).cuda()
        self.transformer = TransformerEncoder(encoder_layer, config.n_layers).cuda()
        self.fc = nn.Linear(config.emb_dim, len(uci_to_int)).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        if x.device != self.device and self.device.type == "cuda":
            x = x.cuda()
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(
            dim=1
        )  # Pooling over the sequence dimension (you can adjust this based on your requirements)
        x = self.fc(x)
        return x

vocab = custom_vocab_in


# Data loading
def get_dataset(fens, vocab):
    dataset = [torch.tensor(fen_to_tensor(fen, vocab)).long() for fen in fens]
    return dataset


def uci_to_tensor(uci_string):
    return torch.tensor(uci_to_int[uci_string]).float()


def run_game(opponent_elo, past, optimizer, model):  # , results, loss_list):
    start_time = time.time()
    board = chess.Board()

    starting_color = random.choice(["white", "black"])
    if starting_color == "white":
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK
    log_probs = []
    attempts = 0
    roundsGame = 0
    earlier_moves = []

    while not board.is_game_over():
        # print('---------------------------------------')
        board_fen = board.fen()
        for earlier_move in earlier_moves:
            board_fen += "%" + earlier_move
        # if len(earlier_moves) < past:
        #    for i in range(past-len(earlier_moves)):
        #        board_fen += "%" + len(board.fen())*"#"

        x_fen = fen_to_tensor(board_fen, max_length).long()

        losses = []
        logits = []
        targets = []

        try:
            best = getBestMove(board, 3000)
        except:
            break

        for i in range(500):
            x_logits = model(x_fen.unsqueeze(0)).squeeze(0)
            try:
                target = uci_to_tensor(best).long().cuda()
                targets.append(target)
                logits.append(x_logits)
            except:
                break
            x_probs = torch.softmax(x_logits, dim=-1)
            del x_logits

            move = torch.multinomial(x_probs, 1)[0]
            log_prob = torch.log(x_probs[move])
            del x_probs

            valid_move = None
            valid_move_uci = valid_move_tokens[move.item()]
            try:
                if chess.Move.from_uci(valid_move_uci) in board.legal_moves:
                    valid_move = chess.Move.from_uci(valid_move_uci)
                    break
            except:
                continue

        if move is None:
            break

        if valid_move is None:
            break
        move = valid_move

        roundsGame += 1
        # print(f"Round: {roundsGame}")
        log_probs.append(log_prob)
        logits = torch.stack(logits, dim=0)
        targets = torch.stack(targets, dim=0)
        if len(targets) < 2:
            logits = logits.squeeze(0)
            targets = targets.squeeze(0)
        loss = criterion(
            logits, targets
        )  # (torch.tensor(logits, requires_grad=True, dtype=torch.float), targets)
        losses.append(loss.item())
        # with model_lock:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        try:
            board.push(move)
        except:
            break

        if past > 0:
            if len(earlier_moves) > past:
                earlier_moves.pop(0)
            earlier_moves.append(board.fen())

        # Opponent move
        opmove = getBestMove(board, opponent_elo)

        while opmove is None:
            opmove = getBestMove(board, opponent_elo)
        board.push(chess.Move.from_uci(opmove))

        if past > 0:
            if len(earlier_moves) > past:
                earlier_moves.pop(0)
            earlier_moves.append(board.fen())

    end_time = time.time()
    game_time = end_time - start_time
    if len(losses) > 0:
        average_losst = sum(losses) / len(losses)  # loss.item() / (roundsGame)
        print(
            f"Loss: {loss.item():.5f}\tLoss/Round: {average_losst:.2f}\tDuration: {game_time:.0f}s\tSeconds/Round {(game_time/roundsGame):.0f}\tRounds: {int(roundsGame)}"
        )


def fen_to_tensor(fen_string, max_length):
    i = 0
    tensor = []

    for s in fen_string:
        if s in custom_vocab_in:
            tensor.append(custom_vocab_in[s])
    return torch.tensor(tensor)


def validate(model, validation_set, loss_f):
    model.eval()
    total_loss = 0.0

    for board_fen in validation_set:
        best_move = getBestMoveFen(board_fen[1], 3500)
        x_fen = fen_to_tensor(board_fen[1], max_length)
        #print(x_fen.shape)
        x_logits = model(x_fen.unsqueeze(0)).squeeze(0)

        try:
            target_actions = (
                uci_to_tensor(best_move).long().cuda()
            )  # [uci_to_tensor(move).long() for move in best_moves]
        except:
            continue
        loss = loss_f(x_logits, target_actions)  # torch.stack(target_actions, dim=0))
        total_loss += loss.item()

    avg_loss = total_loss / len(validation_set)
    print(f"Validation Loss: {avg_loss:.5f}")


# Assume you have a vocab dictionary mapping tokens to indices
vocab_size = len(vocab)
vocab = {token: idx for idx, token in enumerate(vocab)}

# Model, Optimization, and Criterion
model_config = ModelConfig(emb_dim=64, n_layers=8, n_heads=8)
model = ChessTransformer(model_config, vocab_size)
optimizer = optim.AdamW(model.parameters())
criterion = nn.CrossEntropyLoss()

print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

ctx = multiprocessing.get_context("spawn")
# model_lock = ctx.Lock()
PATH = "model.pt"

try:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
except:
    pass


def generate_random_chess_position():
    board = chess.Board()

    # last_fen_1 = None
    # last_fen_2 = None
    gamel = random.randint(4, 30)
    # st = ''
    for _ in range(gamel):
        legal_moves = list(board.legal_moves)
        try:
            random_move = random.choice(legal_moves)
            # last_fen_2 = last_fen_1
            # last_fen_1 = board.fen()
        except:
            return None
        board.push(random_move)
        # try:
        #    st = "%".join([board.fen(), last_fen_1, last_fen_2])
        # except:
        #    st = "%".join([board.fen(), last_fen_1, "#"*len(last_fen_1)])
    st = board.fen()
    return board.fen(), st, board


def getBestMoveFen(board_fen, elo):
    stockfish.reset_engine_parameters()
    stockfish.set_elo_rating(elo)
    stockfish.set_fen_position(board_fen)
    move = None

    for _ in range(10):
        move = stockfish.get_best_move_time(25)
        if move is not None:
            break
    return move


def process_board(board_fens, model, optimizer, criterion):
    if board_fens is None:
        return

    board_fen = board_fens[0]
    boards_fen = board_fens[1]
    board = board_fens[2]
    try:
        best_move = getBestMoveFen(board_fen, 3500)
    except:
        return

    if best_move is None:
        return

    try:
        target_action = uci_to_tensor(best_move).long().cuda()
    except:
        return
    loss_mutliplier = 1
    x_fen = fen_to_tensor(boards_fen, max_length)
    x_logits = model(x_fen.unsqueeze(0)).squeeze(0)
    x_logits = torch.softmax(x_logits, dim=-1)
    move = torch.multinomial(x_logits, 1)[0]
    move = valid_move_tokens[move.item()]
    #try:
    #except:
    #    return

    try:
        if chess.Move.from_uci(move) in board.legal_moves:
            loss_mutliplier = 0.5
    except:
        pass

    loss = criterion(x_logits, target_action)
    loss = loss * loss_mutliplier
    # with model_lock:
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()


def moves_train(
    model, optimizer, num_boards, num_validation_positions, validation_set, criterion
):
    model.train()
    # pool = ctx.Pool(processes=4)  # Adjust the number of processes as needed

    for board in range(num_boards):
        board_fens = generate_random_chess_position()
        # pool.apply_async(process_board, args=(board_fens, model, optimizer))
        process_board(board_fens, model, optimizer, criterion)

        # if board % 50 == 0:
        # avgl = total_loss / (board + 1)
        # print(f"Board {board+1}/{num_boards}")#\tavg loss: {avgl:.5f}")

        # if (board + 1) % num_validation_positions == 0:
        #    validate(model, validation_set)

    # pool.close()
    # pool.join()


def getBestMove(board_n, elo):
    return getBestMoveFen(board_n.fen(), elo)


def generate_validation_set(num_positions):
    validation_set = []

    while len(validation_set) < num_positions:
        gen = generate_random_chess_position()
        if gen != None:
            validation_set.append(gen)

    return validation_set


player_elo = 500
opponent_elo = 2000
num_games = 100
num_validation_positions = 100
num_boards = 100

stockfish.set_elo_rating(opponent_elo)

past = 0
epochs = 2000
num_games = 1

wins = 0
losses = 0
games_played = 0

# Training loop
for epoch in range(10):
    total_loss = 0.0
    rounds = 0
    move = 0
    movensgames = []
    game_results = []
    processes = []
    model.train()
    torch.cuda.empty_cache()
    print("Training on game")
    run_game(opponent_elo, past, optimizer, model)
    print("Training on moves")
    moves_train(
        model,
        optimizer,
        num_boards,
        num_validation_positions,
        generate_validation_set(num_validation_positions),
        criterion,
    )
    print("Validating")
    try:

     validate(model, generate_validation_set(num_validation_positions), criterion)
    except:
        pass

    # Checkpointing
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        PATH,
    )
