import chess

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from dataclasses import dataclass
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import chess.engine
import numpy as np
import string
import time
from stockfish import Stockfish
import multiprocessing
import random

custom_vocab_in = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,  
    'w': 13, 'b': 14,  
    '-': 15,  
    '0': 16, '1': 17, '2': 18, '3': 19, '4': 20, '5': 21, '6': 22, '7': 23, '8': 24, '9': 25,  
    'e': 26, 'c': 27, 'f': 28, 
    '%': 29, '/': 30, ' ': 31, '#': 32, 
    'a': 33, 'd': 34, 'g': 35, 'h': 36
}

custom_vocab_out = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,  # Columns
    '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15,  # Rows
    'q': 16, 'r': 17, 'n': 18
}

custom_vocab_out_inverse = {v: k for k, v in custom_vocab_out.items()}

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ChessGPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, custom_vocab_out, custom_vocab_in):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = True
        self.custom_vocab_out = custom_vocab_out
        self.custom_vocab_in = custom_vocab_in

stockfish = Stockfish("stockfish-path")

embedding_dim = 256
@dataclass
class GPTConfig:
    block_size: int = embedding_dim*2 #1024
    vocab_size: int = len(custom_vocab_in)+2
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = int(embedding_dim)#8) #768
    dropout: float = 0.2
    bias: bool = True
    custom_vocab_in: int = len(custom_vocab_in)+2
    custom_vocab_out: int = len(custom_vocab_out)

class ChessGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Modify the vocabulary size to support UCI notation moves
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.custom_vocab_in, config.n_embd).to(self.device),
            wpe = nn.Embedding(config.block_size, config.n_embd).to(self.device),
            drop = nn.Dropout(config.dropout).to(self.device),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]).to(self.device),
            ln_f = LayerNorm(config.n_embd, bias=config.bias).to(self.device),
        ))

        self.char_generation = nn.Linear(config.n_embd, len(custom_vocab_out)).cuda()
        self.custom_vocab_out = custom_vocab_out

        # Initialize the weights for the custom output vocabulary
        self.char_generation.weight = nn.Parameter(torch.randn(config.custom_vocab_out, config.n_embd).to(self.device))

        # Initialize the embedding matrix with custom input vocabulary
        self.transformer.wte.weight = nn.Parameter(torch.randn(config.custom_vocab_in, config.n_embd).to(self.device))
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    """def forward(self, fen):
        fen = fen.to(self.device).long()
        outputs = self.transformer.wte(fen)

        # Initialize outputs
        for i, block in enumerate(self.transformer.h):
            outputs = block(outputs)

        # Compute the output using the char_generation layer
        logits = self.char_generation(outputs)

        return logits"""
    
    """def forward(self, fen):
        fen = fen.to(self.device).long()
        outputs = self.transformer.wte(fen)

        # Initialize outputs
        for i, block in enumerate(self.transformer.h):
            outputs = block(outputs)

        # Compute the output using the char_generation layer
        logits = self.char_generation(outputs)

        return logits"""

    def forward(self, fen):
        fen = fen.to(self.device).long()
        inputs = self.transformer.wte(fen).squeeze(0)
        start_symbol_index = 0
        stop_symbol_index = 1

        generated_chars = []
        generated_chars_num = 0

        # Start with the start symbol
        #current_char = torch.tensor([start_symbol_index], dtype=torch.long, device='cuda') 
        current_char = 0
        logits = []

        for _ in range(max_length_gen):
            # Apply the character generation linear layer
            char_probs = self.char_generation(inputs)
            logits.append(char_probs)
            char_probs = F.softmax(char_probs, dim=-1)

            # Sample a character from the probability distribution
            current_char_ind = torch.multinomial(char_probs.squeeze(0), 1).squeeze()
            current_char = current_char_ind[0].item()
            # Append the sampled character to the generated sequence
            generated_chars.append(current_char)
            """
            if current_char == stop_symbol_index:
                generated_chars_num += len('</S>')
            elif current_char == start_symbol_index:
                generated_chars_num += len('<S>')
            else:"""
            generated_chars_num += 1

            # Convert the current character to an embedding
            input_us = inputs.unsqueeze(0)
            current_embedding = self.transformer.wte(current_char_ind[0]).unsqueeze(0)
            
            # Concatenate the current character's embedding to the input sequence
            inputs = torch.cat([inputs, current_embedding], dim=0)

            for i, block in enumerate(self.transformer.h):
                inputs = block(input_us)
            inputs = inputs.squeeze(0)
        # Stack the logits for each character along dimension 0
        logits = torch.stack(logits, dim=0)
        
        # Calculate the mean of the logits along the first dimension to get [4, 16]
        logits_mean = logits.mean(dim=1)

        # Return the generated character sequence
        return generated_chars, logits_mean

    """
    def forward(self, fen):
        fen = fen.to(self.device).long()
        inputs = self.transformer.wte(fen)

        outputs = inputs  # Initialize outputs

        for i, block in enumerate(self.transformer.h):
            outputs = block(outputs)

        # Compute the mean of logits along the sequence dimension
        logits = self.uci_head(outputs.mean(dim=1, keepdim=True))

        return logits"""
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def generate(self, fen_input, max_new_tokens, temperature=1.0, top_k=None):
        pass

def uci_to_tensor(uci_string):
    return torch.tensor([custom_vocab_out[c] for c in uci_string]).float()

def fen_to_tensor(fen_string, max_length):
    tensor = [custom_vocab_in[char] for char in fen_string]
    return torch.tensor(tensor, requires_grad=True, dtype=torch.float)


def update_elo_ratings(player_elo, opponent_elo, game_result, change_opp_elo=False, K=32):
    # Calculate expected scores
    expected_player_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    expected_opponent_score = 1 / (1 + 10 ** ((player_elo - opponent_elo) / 400))
    
    # Update Elo ratings
    if game_result == 1:  # Player wins
        actual_player_score = 1.0
        actual_opponent_score = 0.0
    elif game_result == 0.5:  # Draw
        actual_player_score = 0.5
        actual_opponent_score = 0.5
    elif game_result == 0:  # Player loses
        actual_player_score = 0.0
        actual_opponent_score = 1.0

    new_player_elo = player_elo + K * (actual_player_score - expected_player_score)
    new_opponent_elo = opponent_elo + K * (actual_opponent_score - expected_opponent_score)

    if change_opp_elo:
        return new_player_elo, new_opponent_elo
    else:
        return new_player_elo, opponent_elo

cross_entropy = nn.CrossEntropyLoss()

def calc_per_move_loss(best_moves, predictions):
  losses = []
  for i in range(len(best_moves)):
    loss = cross_entropy(predictions[i], best_moves[i])
    losses.append(loss)
  return losses

ctx = multiprocessing.get_context("spawn")
model_lock = ctx.Lock()

max_length = 128
max_length_gen = 4
PYDEVD_DISABLE_FILE_VALIDATION=1

def run_game(opponent_elo, past, optimizer, model, results, loss_list):
    start_time = time.time()
    board = chess.Board()
    
    starting_color = random.choice(['white', 'black'])
    if starting_color == 'white':
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK

    roundsGame = 0
    earlier_moves = []

    best_move_fen = []
    all_logits = []

    tried_list_logits = []
    best_list = []

    while not board.is_game_over():
        #print('---------------------------------------')
        board_fen = board.fen()
        for earlier_move in earlier_moves:
            board_fen += "%" + earlier_move
                    
        x_fen = fen_to_tensor(board_fen, max_length_gen)
        small_tried = []
        try:
            best = getBestMove(board, 3000)
        except:
            break
        for i in range(500):
            generated_chars, all_logits = model(x_fen.unsqueeze(0))
            small_tried.append(all_logits)
            del all_logits
            best_list.append(best)
            move = "".join([custom_vocab_out_inverse[char] for char in generated_chars])
            #print(move)
            val_move = None
            try:
                if chess.Move.from_uci(move) in board.legal_moves:
                    print(move)
                    val_move = chess.Move.from_uci(move)
                    break
            except:
                pass
        tried_list_logits.extend(small_tried)
        del small_tried
        best_move_fen.append(best)      
        if move is None:
            break

        if val_move is None:
            break
        move = val_move

        roundsGame += 1
        print(f"Round: {roundsGame}")
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
          
    if len(best_move_fen) == 0:
        return
    best_t = torch.stack([uci_to_tensor(x) for x in best_list]).long().cuda()
    #print(best_t)
    print(len(tried_list_logits))
    tried_list_logits = torch.stack(tried_list_logits).cuda().view(-1, len(custom_vocab_out))

    best_t = best_t.to(torch.int64).view(-1)  # Convert best_t to int data type

    #best_t = best_t.view(-1)
    loss = nn.CrossEntropyLoss()(tried_list_logits, best_t)
    del tried_list_logits
    del best_t

    if roundsGame > 0:
        loss = loss / roundsGame

    loss_s = 1
    bresult = board.result()

    if bresult == "1-0":
        #print("AI wins")
        if roundsGame > 10:
            loss_s = 0.8 * loss_s
        else:
            loss_s = 0.55 * loss_s
    elif bresult == "0-1":
        #print("AI lost")
        if roundsGame > 10:
            loss_s = 1.2 * loss_s
        else:
            loss_s = 1.5 * loss_s
    loss *= loss_s
    loss_list.append(loss.item())
    with model_lock:
        stats = results['stats']
        stats['games_played'] += 1
        if bresult == "1-0":
            stats['wins'] += 1
        elif bresult == "0-1":
            stats['losses'] += 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    end_time = time.time()
    game_time = end_time - start_time

    if roundsGame == 0:
        average_losst = 0
        avgt = 0
    else:
        average_losst = loss.item() / (roundsGame)
        avgt = game_time/roundsGame
    print(f"Loss: {loss.item():.5f}\tLoss/Round: {average_losst:.2f}\tDuration: {game_time:.0f}s\tSeconds/Round {(avgt):.0f}\tRounds: {int(roundsGame)}")

def generate_random_chess_position():
    board = chess.Board()
    
    last_fen_1 = None
    last_fen_2 = None
    gamel = random.randint(1, 30)
    st = ''
    for _ in range(gamel):
        legal_moves = list(board.legal_moves)
        try:
            random_move = random.choice(legal_moves)
            last_fen_2 = last_fen_1
            last_fen_1 = board.fen()
        except:
            return None
        board.push(random_move)

        if last_fen_1 != None and last_fen_2 != None:
            st = "%".join([board.fen(), last_fen_1, last_fen_2])
        elif last_fen_1 != None and last_fen_2 == None:
            st = "%".join([board.fen(), last_fen_1])
        else:
            st = board.fen()
    return board.fen(), st, board

def getBestMove(board_n, elo):
    return getBestMoveFen(board_n.fen(), elo)

def generate_validation_set(num_positions):
    validation_set = []
    print("Generating validation set")
    while len(validation_set) < num_positions:
        gen = generate_random_chess_position()
        if gen != None:
            validation_set.append(gen)
    print("Done generating validation set")
    return validation_set

def getBestMoveFen(board_fen, elo):
    stockfish.reset_engine_parameters()
    stockfish.set_elo_rating(elo)
    stockfish.set_fen_position(board_fen)
    move = None
    attempts = 0
    while move is None:
        attempts += 1
        if attempts > 5:
            return None
        move =  stockfish.get_best_move_time(25)
    return move
    try:
        
        return
    except:
        return None

def validate(model, validation_set):
    model.eval()
    total_loss = 0.0
    i = 0
    for board_fen, board_fens, _ in validation_set:
        best_move = getBestMoveFen(board_fen, 3500)
        if best_move is None:
            continue
        x_fen = fen_to_tensor(board_fens, max_length)
        
        try:
            _, x_logits = model(x_fen.unsqueeze(0))
            target_actions = uci_to_tensor(best_move).long().cuda()
            loss = nn.CrossEntropyLoss()(x_logits, target_actions)
            total_loss += loss.item()
            i += 1
            if i % 50 == 0:
                print(f"Validation Loss: {loss.item():.5f}")
        except:
            print("Error in validate")
            print(len(x_logits))
            print(len(target_actions))
            continue          

    avg_loss = total_loss / len(validation_set)
    print(f"Validation Loss: {avg_loss:.5f}")

def process_board(board_fens, model, optimizer, validation_set, ind, num_validation_positions):
    if board_fens is None:
        return

    board_fen = board_fens[0]
    boards_fen = board_fens[1]
    board = board_fens[2]
    best_move = getBestMoveFen(board_fen, 3500)

    if best_move is None:
        return
    loss_mutliplier = 1
    target_action = uci_to_tensor(best_move).long().cuda()
    x_fen = fen_to_tensor(boards_fen, max_length)
    generated_chars, x_logits = model(x_fen.unsqueeze(0))
    move = "".join([custom_vocab_out_inverse[char] for char in generated_chars])
    if board.is_legal(chess.Move.from_uci(move)):
        loss_mutliplier = 0.5
    
    loss = nn.CrossEntropyLoss()(x_logits, target_action)
    loss = loss * loss_mutliplier

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def moves_train(model, optimizer, num_boards, num_validation_positions, validation_set):
    model.train()
    print("Starting training on random positions")
    with ctx.Pool(processes=4) as pool:
        for board in range(num_boards):
            if board % 50 == 0:
                print(f"Board {board+1}/{num_boards}")
            board_fens = generate_random_chess_position()
            pool.apply_async(process_board, args=(board_fens, model, optimizer, validation_set, board, num_validation_positions))

        print("Waiting for processes to finish")

        pool.close()
        pool.join()

        validate(model, validation_set)
        print("Done validating")

if __name__ == '__main__':

    epochs = 100
    batch_size = 64
    
    model = ChessGPT(GPTConfig)
    model.train()
    model.share_memory()

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    try:
        checkpoint = torch.load('save-path')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("No checkpoint found")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Using GPU")

    player_elo = 500
    opponent_elo = 2000
    num_games = 100  
    num_validation_positions = 100
    num_boards=200

    stockfish.set_elo_rating(opponent_elo)

    past = 0
    epochs = 2000
    num_games = 3

    wins = 0
    losses = 0
    games_played = 0
    scaler = GradScaler()
    stats_store = {'wins': 0, 'losses': 0, 'games_played': 0}
    for epoch in range(epochs):
        total_loss = 0.0
        rounds = 0
        move = 0
        total_loss = 0.0
        movensgames = []
        game_results = []
        processes = []
        model.train()
        torch.cuda.empty_cache()

        with ctx.Manager() as manager:
            results = manager.dict()
            loss = manager.list()
            results['stats'] = manager.dict(wins=0, losses=0, games_played=0)
            for game in range(num_games):
                game_process = ctx.Process(target=run_game, args=(opponent_elo, past, optimizer, model, results, loss))
                game_process.start()
                processes.append(game_process)
            
            for game_process in processes:
                game_process.join()
            
            stats = results['stats']
            wins = stats['wins']
            losses = stats['losses']
            loss = sum(loss)
            games_played = stats['games_played']

            stats_store['wins'] += wins
            stats_store['losses'] += losses
            stats_store['games_played'] += games_played

            print(f"Epoch: {epoch}\tWins: {stats_store['wins']}\tLosses: {stats_store['losses']}\tGames played: {stats_store['games_played']}\tLoss: {loss:.5f}")
            moves_train(model, optimizer, num_boards, num_validation_positions, generate_validation_set(num_validation_positions))
        
        lr_scheduler.step()
        if epoch % 4 == 0:
            print("Saving checkpoint")
            torch.save({
                'model_state_dict': model.state_dict(),  # Save the model's weights
                'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state
            }, 'save-path')
            print("Checkpoint saved")

    torch.save({
        'model_state_dict': model.state_dict(),  # Save the model's weights
        'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state
    }, 'save-path')
