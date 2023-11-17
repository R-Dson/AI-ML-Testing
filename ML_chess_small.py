import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import string
import chess
import chess.engine
import time
from stockfish import Stockfish
import random
import numpy as np
import math
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter


valid_move_tokens = [
    f"{c1}{r}{c2}{r2}"
    for c1 in string.ascii_lowercase[:8]
    for r in string.digits[1:9]
    for c2 in string.ascii_lowercase[:8]
    for r2 in string.digits[1:9]
]

uci_to_int = {uci_move: i for i, uci_move in enumerate(valid_move_tokens)}

custom_vocab_in = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14
}

max_length = 128


class ModelConfig:
    def __init__(self, emb_dim=64, n_layers=8, n_heads=8, dropout=0.4):
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(500000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.to(x.dtype)
        return x + pe[:, :x.size(1)].detach()

class ChessTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, config.emb_dim).cuda()
        nn.init.xavier_uniform_(self.embedding.weight)
        self.positional_encoding = PositionalEncoding(config.emb_dim).cuda()
        #self.norm = nn.LayerNorm(config.emb_dim).cuda()
        #nn.init.constant_(self.norm.weight, 1.0)
        #nn.init.constant_(self.norm.bias, 0.0)

        encoder_layer = TransformerEncoderLayer(config.emb_dim, config.n_heads).cuda()
        self.transformer = TransformerEncoder(encoder_layer, config.n_layers).cuda()
        self.dropout = nn.Dropout(p=config.dropout).cuda()
        self.fc = nn.Linear(config.emb_dim, len(uci_to_int)).cuda()
        nn.init.xavier_uniform_(self.fc.weight)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        if x.device != self.device and self.device.type == "cuda":
            x = x.cuda()

        xe = self.embedding(x.long())
        xp = self.positional_encoding(xe)
        xn = nn.functional.layer_norm(xp, normalized_shape=xp.size()[1:])
        #x = self.norm(x)
        xt = self.transformer(xn)
        xm = xt.mean(dim=1)  # Pooling over the sequence dimension (you can adjust this based on your requirements)
        xn = nn.functional.layer_norm(xm, normalized_shape=xm.size()[1:]) 
        xd = self.dropout(xn)
        xf = self.fc(xd).squeeze(0)
        if torch.isnan(xf).any():
            print(self.embedding.weight)
            print(xf)
        return xf

vocab = custom_vocab_in

# Data loading
def get_dataset(fens, vocab):
    dataset = [torch.tensor(fen_to_tensor(fen, vocab)).long() for fen in fens]
    return dataset


def uci_to_tensor(uci_string):
    return torch.tensor(uci_to_int[uci_string]).float()


def run_game(opponent_elo, past, optimizer, model, epoch, criterion, process_id):
    start_time = time.time()
    board = chess.Board()
    stockfish = Stockfish(
        "stockfish-path"
    )
    stockfish.set_elo_rating(opponent_elo)
    
    if process_id == 4:
        writer_loss = SummaryWriter(f'model-path/runs/{process_id}-loss')
    writer_rounds = SummaryWriter(f'model-path/runs/{process_id}-rounds')

    starting_color = random.choice(["white", "black"])
    starting_bool = False
    if starting_color == "white":
        board.turn = chess.WHITE
        starting_bool = chess.WHITE
    else:
        board.turn = chess.BLACK
        starting_bool = chess.BLACK

    log_probs = []
    roundsGame = 0
    earlier_moves = []
    losses = []
    num_perfect = 0
    valid_Moves = 0
    
    board_history = []

    while not board.is_game_over():
        # print('---------------------------------------')
        board_fen = board.fen()
        for earlier_move in earlier_moves:
            board_fen += "%" + earlier_move

        color = int(board.turn)

        x_fen = ([piece_at(square, board) for square in chess.SQUARES])
        x_fen.insert(0, color)

        best_num = 0
        batch = 20
        tmp_hist_curr = board_history.copy()
        tmp_hist_curr = tmp_hist_curr[:7]
        tmp_hist_curr.insert(0, x_fen)
        
        tmp_hist_curr = torch.LongTensor(tmp_hist_curr)
        tmp_hist_curr = tmp_hist_curr.view(-1).unsqueeze(0)

        best = getBestMove(stockfish, board, 3000)
        try:
            pass
        except:
            break
        losses_sum = 0
        loss_i = 0
        losses_batch = []
        good_moves = []
        best_moves = []
        for i in range(batch):
            x_logits = model(tmp_hist_curr).squeeze(0)
            
            #x_logits_normalized = nn.functional.normalize(x_logits, p=2, dim=-1)
            x_probs = torch.softmax(x_logits, dim=-1)
            move = torch.multinomial(x_probs, 1)[0]
            log_prob = torch.log(x_probs[move])
            try:
                target = uci_to_tensor(best).long().cuda()
            except:
                break
            optimizer.zero_grad()
            loss = criterion(x_logits, target)
            
            try:
                if chess.Move.from_uci(valid_move_uci) in board.legal_moves:
                    loss = 0.75*loss
            except:
                pass
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
            optimizer.step()

            losses_sum += loss.item()
            loss_i += 1
            #loss = criterion(x_logits, target)
            #losses_batch.append((x_logits.clone(), target.clone(), i))
            
            del x_logits            
            del x_probs

            valid_move = None
            valid_move_uci = valid_move_tokens[move.item()]
            try:
                if chess.Move.from_uci(valid_move_uci) in board.legal_moves:
                    valid_move = chess.Move.from_uci(valid_move_uci)
                    valid_Moves += 1
                    good_moves.append(i)
                    if valid_move_uci == best:
                        num_perfect += 1
                        best_num += 1
                        best_moves.append(i)
                    break
            except:
                continue
        del losses_batch
        if valid_move is None:
            break
        #valid_move

        roundsGame += 1
        log_probs.append(log_prob)
        del log_prob
        if loss_i > 0:
            losses.append(losses_sum/loss_i)
        board_history.insert(0, x_fen)

        try:
            board.push(valid_move)
        except:
            break
        #print(f"AI Color: {starting_color}")
        #print(board)

        if past > 0:
            if len(earlier_moves) > past:
                earlier_moves.pop(0)
            earlier_moves.append(board.fen())

        # Opponent move
        opmove = getBestMove(stockfish, board, opponent_elo)

        while opmove is None:
            opmove = getBestMove(stockfish, board, opponent_elo)
        
        x_fen = ([piece_at(square, board) for square in chess.SQUARES])
        x_fen.insert(0, int(board.turn))
        board_history.insert(0,x_fen)#([piece_at(square, board) for square in chess.SQUARES])
        board.push(chess.Move.from_uci(opmove))
        #print(board)
        if past > 0:
            if len(earlier_moves) > past:
                earlier_moves.pop(0)
            earlier_moves.append(board.fen())

    won = False
    if board.is_checkmate() and board.turn == starting_bool:
        won = True

    end_time = time.time()
    game_time = end_time - start_time
    if len(losses) > 0:
        average_losst = sum(losses) / len(losses)
        if valid_Moves > 0:
            avg_perfect = num_perfect / valid_Moves
        else: 
            avg_perfect = 0
        print(
            f"Avg loss: {average_losst:.2f}\t best moves ratio: {avg_perfect:.2f}, best moves: {num_perfect}, valid moves: {valid_Moves} \tDuration: {game_time:.0f}s\tSeconds/Round {(game_time/roundsGame):.0f}\tRounds: {int(roundsGame)}"
        )
    
    if process_id == 4:
        if len(losses) > 0:
            writer_loss.add_scalar("loss x epoch (game)", sum(losses)/len(losses), epoch)
        writer_loss.close()
        
    writer_rounds.add_scalar("rounds x epoch (game)", roundsGame, epoch)
    writer_rounds.close()
    return losses, won


def fen_to_tensor(fen_string, max_length):
    i = 0
    tensor = []

    for s in fen_string:
        if s in custom_vocab_in:
            tensor.append(custom_vocab_in[s])
    return torch.tensor(tensor)


def piece_at(square, board):
    piece = board.piece_at(square)
    if piece is None:
        return 2
    else:
        color_factor = 1 if piece.color == chess.WHITE else 2
        return color_factor * piece.piece_type+2


def validate(model, validation_set, loss_f, stockfish):
    model.eval()
    total_loss = 0.0

    for board_fen in validation_set:
        board = board_fen[2]
        best_move = getBestMoveFen(stockfish, board_fen[1], 3500)
        x_fen = ([piece_at(square, board) for square in chess.SQUARES])
        color = int(board.turn)
        x_fen.insert(0, color)
        x_fen = torch.tensor([x_fen])
        #color = torch.tensor([color], dtype=torch.long)
        x_logits = model(x_fen).squeeze(0)

        try:
            target_actions = uci_to_tensor(best_move).long().cuda()
        except:
            continue
        loss = loss_f(x_logits, target_actions)
        total_loss += loss.item()
        del loss
    avg_loss = total_loss / len(validation_set)
    print(f"Validation Loss: {avg_loss:.5f}")
    return avg_loss


vocab_size = len(vocab)
vocab = {token: idx for idx, token in enumerate(vocab)}



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

def getBestMoveFen(stockfish, board_fen, elo):
    stockfish.reset_engine_parameters()
    stockfish.set_elo_rating(elo)
    stockfish.set_fen_position(board_fen)
    move = None

    for _ in range(10):
        move = stockfish.get_best_move_time(25)
        if move is not None:
            break
    return move


def process_board(board_fens, model, optimizer, criterion, stockfish):
    if board_fens is None:
        return 0, 0

    board_fen = board_fens[0]
    boards_fen = board_fens[1]
    board = board_fens[2]
    try:
        best_move = getBestMoveFen(stockfish, board_fen, 3500)
        target_action = uci_to_tensor(best_move).long().cuda()
    except:
        return 0, 0

    loss_mutliplier = 1
    x_fen = ([piece_at(square, board) for square in chess.SQUARES])

    color = int(board.turn)
    x_fen.insert(0, color)
    x_fen = torch.tensor([x_fen])
    #color = torch.tensor(color, dtype=torch.long)
    
    x_logits = model(x_fen).squeeze(0)
    del x_fen
    x_logits = torch.softmax(x_logits, dim=-1)
    move = torch.multinomial(x_logits, 1)[0]
    move = valid_move_tokens[move.item()]

    bestmove = 0

    try:
        if chess.Move.from_uci(move) in board.legal_moves:
            loss_mutliplier = 0.5
            if move == best_move:
                loss_mutliplier = 0.1
                bestmove = 1
    except:
        pass

    loss = criterion(x_logits, target_action)
    loss = loss * loss_mutliplier
    # with model_lock:
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 50)
    optimizer.step()
    del x_logits
    return bestmove, loss.item()


def moves_train(
    model, optimizer, num_boards, num_validation_positions, validation_set, criterion
):
    model.train()
    # pool = ctx.Pool(processes=4)  # Adjust the number of processes as needed
    all_losses = []
    num_best_moves = 0
    for board in range(num_boards):
        board_fens = generate_random_chess_position()
        # pool.apply_async(process_board, args=(board_fens, model, optimizer))
        best, losses = process_board(board_fens, model, optimizer, criterion)
        all_losses.append(losses)
        num_best_moves += best

        # if board % 50 == 0:
        # avgl = total_loss / (board + 1)
        # print(f"Board {board+1}/{num_boards}")#\tavg loss: {avgl:.5f}")

        # if (board + 1) % num_validation_positions == 0:
        #    validate(model, validation_set)
    print(f"Best moves: {num_best_moves}/{num_boards}")
    return all_losses
    # pool.close()
    # pool.join()


def getBestMove(stockfish, board_n, elo):
    return getBestMoveFen(stockfish, board_n.fen(), elo)


def generate_validation_set(num_positions):
    validation_set = []

    while len(validation_set) < num_positions:
        gen = generate_random_chess_position()
        if gen != None:
            validation_set.append(gen)

    return validation_set


player_elo = 500
opponent_elo = 2000
#num_games = 100
num_validation_positions = 10
num_boards = 10


past = 0
epochs = 20000

wins = 0
losses = 0
games_played = 0


# Training loop
def main(model, optimizer, scheduler, criterion):

    wins = 0
    num_processes = 8
    processes = []
    
    for epoch in range(epochs):
        #print("Training on game")
        model.train()
        opp_elo = random.randint(1, opponent_elo)
        #losses_run, won = run_game(opp_elo, past, optimizer, model)
        #if won:
        #    wins += 1
        
        print(f'wins: {wins} / {epoch+1}, Opp elo: {opp_elo}')
        for process_id in range(num_processes):
            opp_elo = random.randint(1, opponent_elo)
            process = mp.Process(target=run_game, args=(opp_elo, past, optimizer, model, epoch, criterion, process_id))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        
        #print("Training on moves")
        """
        if epoch % 3 == 0:
            losses_moves = moves_train(
                model,
                optimizer,
                num_boards,
                num_validation_positions,
                generate_validation_set(num_validation_positions),
                criterion,
            )
            if losses_moves != None and len(losses_moves) > 0:
                writer.add_scalar("loss x epoch (moves)", sum(losses_moves)/len(losses_moves), epoch)
        torch.cuda.empty_cache"""
        #print("Validating")
        #val_loss = validate(model, generate_validation_set(num_validation_positions), criterion)
        #writer.add_scalar("val loss (move)", val_loss)

        #print("Saving")
        # Checkpointing
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            PATH,
        )

        scheduler.step()
        processes.clear()
    

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    model_config = ModelConfig(emb_dim=64, n_layers=8, n_heads=32)
    model = ChessTransformer(model_config, vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    PATH = "model-path"
    try:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except:
        pass

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.share_memory()
    
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    
    criterion = nn.CrossEntropyLoss()
    criterion.share_memory()

    main(model, optimizer, scheduler, criterion)
    
