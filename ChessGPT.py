
# Inspiration from https://github.com/karpathy/nanoGPT

import chess
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
import torch.optim as optim
import chess.engine
from torch.autograd import Variable
import numpy as np
from chess.engine import SimpleEngine


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

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import string

valid_move_tokens = [f"{c1}{r}{c2}{r2}" for c1 in string.ascii_lowercase[:8] 
                          for r in string.digits[1:8]
                          for c2 in string.ascii_lowercase[:8]
                          for r2 in string.digits[1:8]]

custom_vocab = {
    " ": 0, "-": 1, "K": 2, "Q": 3, "R": 4, "N": 5, "B": 6, "P": 7,
    "k": 8, "q": 9, "r": 10, "n": 11, "b": 12, "p": 13, 
    ".": 14, "/": 15,
    "1": 16, "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23,
    "w": 24, "W": 25, "0": 26, "a": 27, "b": 28, "c": 29, "d": 30, "e": 31, "f": 32, "g": 33, "h": 34,
    "A": 35, "B": 36, "C": 37, "D": 38, "E": 39, "F": 40, "G": 41, "H": 42,
}

@dataclass
class GPTConfig:
    block_size: int = 256 #1024
    vocab_size: int = len(valid_move_tokens)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256 #768
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
        # self.transformer = self.transformer.to(self.device)

        # Output layer for UCI notation
        self.uci_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(self.device)
        self.transformer.wte.weight = self.uci_head.weight

         # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, fen):
        fen = fen.to(self.device).long()
        inputs = self.transformer.wte(fen)
        
        outputs = inputs  # Initialize outputs
        for i, block in enumerate(self.transformer.h):
            outputs = block(outputs)
        
        logits = self.uci_head(outputs)
        
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

embedding_dim = 32
embedding_matrix = np.random.rand(len(custom_vocab), embedding_dim)

def fen_to_tensor(fen_string):
    tokens = list(fen_string)
    token_indices = [custom_vocab[token] for token in tokens]
    embeddings = [embedding_matrix[index] for index in token_indices]

    # convert the list of numpy arrays into a single numpy array
    embeddings = np.array(embeddings)

    tensor = torch.tensor(embeddings)

    sequence_length = 64
    if tensor.size(0) < sequence_length:
        padding = torch.zeros(sequence_length - tensor.size(0), embedding_dim)
        tensor = torch.cat((tensor, padding))
    elif tensor.size(0) > sequence_length:
        tensor = tensor[:sequence_length, :]
        
    return tensor

from stockfish import Stockfish

model = ChessGPT(GPTConfig)
optimizer = optim.Adam(model.parameters(), lr=6e-4, weight_decay=0.01, betas=(0.9, 0.95))

try:
    checkpoint = torch.load('model_checkpoint.pth')

    # Load model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
except:
    print("No checkpoint found")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("Using GPU")

model.to(device)


epochs = 100
stockfish = Stockfish("stockfish-path-here")
penalty = 1.0

totalRounds = []

for epoch in range(epochs):
    total_loss = 0.0
    
    # Initialize chess board
    board = chess.Board()
    attemts = 0
    failedAttempts = 0

    stockfish.reset_engine_parameters()
    stockfish.set_fen_position(board.fen())
    best_move = stockfish.get_best_move_time(10)

    rounds = 0
    move = 0
    
    while not board.is_game_over():
        oldBoard = board.copy()
        
        try:
            input_tensor = fen_to_tensor(board.fen())
        except:
            # Todo: fix?
            print("Failed to convert fen to tensor")
            print('---------------------------------------')
            print('---------------------------------------')
            print('---------------------------------------')
            print(board)
            print('---------------------------------------')
            print('---------------------------------------')
            print('---------------------------------------')
            break
            
        output = model(input_tensor)

        output = output.view(64, 32, -1)  # Reshape the output
        temperature = 0.7  # Adjust the temperature
        output = output / temperature

        # Applying softmax to get probabilities
        probabilities = torch.softmax(output, dim=-1)

        probabilities = probabilities.view(-1, len(valid_move_tokens))

        # Sample move index
        max_prob_indices = torch.argmax(probabilities, dim=-1)

        # Get the highest probability move index
        max_prob_index = max_prob_indices[0].item()

        # Get the corresponding move
        move_predict_uci = valid_move_tokens[max_prob_index]

        loss = 0
        legalmove = False

        # fails when the move is the same square
        try: 
            legalmove = oldBoard.is_legal(chess.Move.from_uci(move_predict_uci))
        except:
            loss += 5
             
        attemts += 1
        if legalmove:
            rounds += 1
            board.push(chess.Move.from_uci(move_predict_uci))
        else:
            loss += penalty
            failedAttempts += 1
        
        try:
            target_predict = fen_to_tensor(board.fen())  # Convert the predicted move board to tensor
        except:
            pass
        oldBoard.push(chess.Move.from_uci(best_move))
        target = fen_to_tensor(oldBoard.fen())  # Convert the best move to tensor
        oldBoard.pop()

        loss += nn.CrossEntropyLoss()(target_predict.float(), target.float())
        loss = Variable(loss, requires_grad = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if legalmove:
            move += 1
            print(move)
            print(board)
            print("---------------------------------------")

            # opponent move
            stockfish.reset_engine_parameters()
            stockfish.set_fen_position(board.fen())
            emove = stockfish.get_best_move_time(100)
            board.push(chess.Move.from_uci(emove))
            move += 1
            print(move)
            print(board)
            print("---------------------------------------")
            if board.is_game_over():
                print("Game over")
                break
            
            # New best Move
            stockfish.set_fen_position(board.fen())
            best_move = stockfish.get_best_move_time(100)
            

        total_loss += loss.item()
        if attemts % 1000 == 0 and attemts != 0:
            average_loss = total_loss / attemts
            print(f"Attempts: {attemts}\tAverage loss/attempt: {average_loss:.5f} \t Failed attempts: {failedAttempts} \tMove: {move_predict_uci}\tBest move: {best_move}")
        if attemts > 2000:
            print("To many attemts, resetting board")
            break
    totalRounds.append(rounds)
    avgl = -1
    if rounds != 0:
        avgl = total_loss/rounds
    print(f"Total Epoch {epoch + 1}\t Loss/round: {avgl:.5f}\tRounds: {rounds}")
print(totalRounds)
torch.save({
    'model_state_dict': model.state_dict(),  # Save the model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer state
}, 'model_checkpoint.pth')
