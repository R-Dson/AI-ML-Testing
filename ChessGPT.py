
# Inspimport chess

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
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = True


valid_move_tokens = [f"{c1}{r}{c2}{r2}" for c1 in string.ascii_lowercase[:8] 
                          for r in string.digits[1:9]
                          for c2 in string.ascii_lowercase[:8]
                          for r2 in string.digits[1:9]]

uci_to_int = {uci_move: i for i, uci_move in enumerate(valid_move_tokens)}

custom_vocab = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # Uppercase letters for white pieces
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,  # Lowercase letters for black pieces
    'w': 13, 'b': 14,  # Side to move ('w' for White, 'b' for Black)
    '-': 15,  # No castling ability
    'a1': 16, 'b1': 17, 'c1': 18, 'd1': 19, 'e1': 20, 'f1': 21, 'g1': 22, 'h1': 23,  # Square a1 to h1
    'a2': 24, 'b2': 25, 'c2': 26, 'd2': 27, 'e2': 28, 'f2': 29, 'g2': 30, 'h2': 31,  # Square a2 to h2
    'a3': 32, 'b3': 33, 'c3': 34, 'd3': 35, 'e3': 36, 'f3': 37, 'g3': 38, 'h3': 39,  # Square a3 to h3
    'a4': 40, 'b4': 41, 'c4': 42, 'd4': 43, 'e4': 44, 'f4': 45, 'g4': 46, 'h4': 47,  # Square a4 to h4
    'a5': 48, 'b5': 49, 'c5': 50, 'd5': 51, 'e5': 52, 'f5': 53, 'g5': 54, 'h5': 55,  # Square a5 to h5
    'a6': 56, 'b6': 57, 'c6': 58, 'd6': 59, 'e6': 60, 'f6': 61, 'g6': 62, 'h6': 63,  # Square a6 to h6
    'a7': 64, 'b7': 65, 'c7': 66, 'd7': 67, 'e7': 68, 'f7': 69, 'g7': 70, 'h7': 71,  # Square a7 to h7
    'a8': 72, 'b8': 73, 'c8': 74, 'd8': 75, 'e8': 76, 'f8': 77, 'g8': 78, 'h8': 79,  # Square a8 to h8
    '0': 80, '1': 81, '2': 82, '3': 83, '4': 84, '5': 85, '6': 86, '7': 87, '8': 88, '9': 89,  # Digits for halfmove clock and fullmove counter
    '%': 90, '/': 91, ' ': 92, '#': 93  # Padding
}

embedding_dim = 256

stockfish = Stockfish("stockfish-path")

@dataclass
class GPTConfig:
    block_size: int = embedding_dim #1024
    vocab_size: int = len(valid_move_tokens)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = embedding_dim #768
    dropout: float = 0.2
    bias: bool = True

class ChessGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Modify the vocabulary size to support UCI notation moves
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd).to(self.device),
            wpe = nn.Embedding(config.block_size, config.n_embd).to(self.device),
            drop = nn.Dropout(config.dropout).to(self.device),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]).to(self.device),
            ln_f = LayerNorm(config.n_embd, bias=config.bias).to(self.device),
        ))

        self.uci_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(self.device)
        
        self.transformer.wte.weight = self.uci_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # report number of parameters
        #print("number of parameters: %.2fM" % (self.get_num_params()/1e6,)
    
    def forward(self, fen):
        fen = fen.to(self.device).long()
        inputs = self.transformer.wte(fen)

        outputs = inputs  # Initialize outputs
        for i, block in enumerate(self.transformer.h):
            outputs = block(outputs)

        # Compute the mean of logits along the sequence dimension
        logits = self.uci_head(outputs.mean(dim=1, keepdim=True))

        return logits
    
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
    return torch.tensor([uci_to_int[uci_string]]).float()

def fen_to_tensor(fen_string, max_length):
    i = 0
    tensor = []

    while i < len(fen_string):
        char = fen_string[i]
        if char in custom_vocab:
            tensor.append(custom_vocab[char])
            i += 1
        elif char.isalpha() and i + 1 < len(fen_string) and fen_string[i:i+2] in custom_vocab:
            tensor.append(custom_vocab[fen_string[i:i+2]])
            i += 2
    if max_length != 0:
        if len(tensor) < max_length:
            padding = [0] * (max_length - len(tensor))
            tensor += padding
        else:
            tensor = tensor[:max_length]

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

max_length = 192

def run_game(opponent_elo, past, optimizer, central_model, wins, losses, games_played):
    start_time = time.time()
    board = chess.Board()
    
    starting_color = random.choice(['white', 'black'])
    if starting_color == 'white':
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK
    log_probs = []
    attempts = 0
    roundsGame = 0
    earlier_moves = []

    best_move_fen = []
    all_logits = []

    while not board.is_game_over():
        #print('---------------------------------------')
        board_fen = board.fen()
        for earlier_move in earlier_moves:
            board_fen += "%" + earlier_move
        if len(earlier_moves) < past:
            for i in range(past-len(earlier_moves)):
                board_fen += "%" + len(board.fen())*"#"
                    
        x_fen = fen_to_tensor(board_fen, max_length)
        x_logits_og = central_model(x_fen.unsqueeze(0)).squeeze(0)
        x_logits = torch.softmax(x_logits_og, dim=1)

        with torch.no_grad():
            action = torch.multinomial(x_logits, 1)[0]
            log_prob = torch.log(x_logits[0][action])

        move = valid_move_tokens[action.item()]
        try:
            move = chess.Move.from_uci(move)
            if move in board.legal_moves:
                all_logits.append(x_logits_og.squeeze(0))
        except:
            move = None
            
        while move is None or move not in board.legal_moves:
            attempts += 1
            if attempts > 10000:
                move = None
                break

            x_logits_og = central_model(x_fen.unsqueeze(0)).squeeze(0)
            x_logits = torch.softmax(x_logits_og, dim=1)

            with torch.no_grad():
                action = torch.multinomial(x_logits, 1)[0]
                log_prob = torch.log(x_logits[0][action])
                del x_logits
                    
            move = valid_move_tokens[action.item()]
            try:
                move = chess.Move.from_uci(move)
                if move in board.legal_moves:
                    all_logits.append(x_logits_og.squeeze(0))
            except:
                move = None

        best = getBestMove(board, 3000)
        if move is None:
            break

        roundsGame += 1
        log_probs.append(log_prob)
            
        board.push(move)
        best_move_fen.append(best)

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
    
    best_t = torch.tensor([uci_to_tensor(best).long().cuda() for best in best_move_fen]).cuda()
    if len(all_logits) == 0:
        return
    all_logits_t = torch.stack(all_logits, dim=0) 
    loss = nn.CrossEntropyLoss()(all_logits_t, best_t)

    loss_s = 1
    bresult = board.result()
    if bresult == "1-0":
        print("AI wins")
        if roundsGame > 10:
            loss_s = 0.8 * loss_s
        else:
            loss_s = 0.55 * loss_s
    elif bresult == "0-1":
        print("AI lost")
        if roundsGame > 10:
            loss_s = 1.2 * loss_s
        else:
            loss_s = 1.5 * loss_s
    else:
        if roundsGame < 10:
            loss_s = 0.8 * loss_s
    loss *= loss_s

    with model_lock:
        games_played += 1
        if bresult == "1-0":
            wins += 1
        elif bresult == "0-1":
            losses += 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(central_model.parameters(), 0.5)
        optimizer.step()

    end_time = time.time()
    game_time = end_time - start_time

    average_losst = loss.item() / (roundsGame)
    print(f"Loss: {loss.item():.5f}\tLoss/Round: {average_losst:.2f}\tDuration: {game_time:.0f}s\tSeconds/Round {(game_time/roundsGame):.0f}\tRounds: {int(roundsGame)}")

def getBestMove(board_n, elo):
    stockfish.reset_engine_parameters()
    stockfish.set_elo_rating(elo)
    stockfish.set_fen_position(board_n.fen())
    return stockfish.get_best_move_time(25)

if __name__ == '__main__':
    
    model = ChessGPT(GPTConfig)
    model.train()
    model.share_memory()

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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
    opponent_elo = 1500

    stockfish.set_elo_rating(opponent_elo)

    past = 3
    epochs = 200
    num_games = 5

    wins = 0
    losses = 0
    games_played = 0
    scaler = GradScaler()
    model.train()
    wins, losses, games_played = 0, 0, 0
    for epoch in range(epochs):
        total_loss = 0.0
        rounds = 0
        move = 0
        total_loss = 0.0
        movensgames = []
        game_results = []
        processes = []
        
        for game in range(num_games):
            game_process = ctx.Process(target=run_game, args=(opponent_elo, past, optimizer, model, wins, losses, games_played))
            game_process.start()
            processes.append(game_process)

        for game_process in processes:
            game_process.join()

        lr_scheduler.step()
        torch.cuda.empty_cache()

        print(f"Wins: {wins}\tLosses: {losses}\tGames played: {games_played}")
        
        if epoch % 4 == 0:
            print("Saving checkpoint")
            torch.save({
                'model_state_dict': model.state_dict(),  # Save the model's weights
                'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state
            }, 'save-path')

    torch.save({
        'model_state_dict': model.state_dict(),  # Save the model's weights
        'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state
    }, 'save-path')
