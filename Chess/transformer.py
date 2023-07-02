from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
import math
import transformers
from transformers import Trainer, TrainingArguments
import sklearn
from sklearn import metrics
from transformers import EarlyStoppingCallback
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization()
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

    def trainModel(self, train_loader, validation_loader, num_epochs, batch_size, lr=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', device)

        if hasattr(self, 'optimizer') is False:
            print('optimizer')
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-9)

        if hasattr(self, 'criterion') is False:
            print('criterion')
            self.criterion = nn.CrossEntropyLoss()

        self.train()
        best_loss = -float('inf')
        patience = 12
        best_epoch = 0
        trl = len(train_loader)
        num_batches = (trl - 1) // batch_size + 1
        print('num_batches', num_batches)
        self.cuda()

        print('starting training loop')
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            bi = 0
            for batch_inputs, batch_labels in train_loader:
                batch_labels = torch.tensor(batch_labels)
                batch_inputs = torch.tensor(batch_inputs)
                bi += 1

                with torch.cuda.amp.autocast():
                    output = self(batch_inputs.to('cuda').float())
                    loss = self.criterion(output, batch_labels.to('cuda').float())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if bi % 1000 == 0:
                    print(f"Epoch {bi}/{trl}")

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches

            loss, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
            if validation_loader is not None:
                loss, accuracy, precision, recall, f1 = self.evaluate3(validation_loader)
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, loss: {loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    def evaluate2(self, loader):
        total_loss = 0
        self.eval()
        #criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            #predictions = []
            #targets = []
            for inputs, labels in loader:
                outputs = self(inputs)
                #_, predicted = torch.max(outputs, dim=1)
                #predictions.extend(predicted.tolist())
                #targets.extend(labels.tolist())
                labels = labels.cuda()
                loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()
        #total_samples = len(targets)
        #correct_predictions = sum(pred == target for pred, target in zip(predictions, targets))
        #accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss
    
    def evaluate3(self, loader):
        total_loss = 0
        predictions = []
        targets = []
        self.eval()
        # Disable the warning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self(inputs)
                labels = labels.cuda()
                loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()

                predicted_labels = torch.round(torch.sigmoid(outputs)).cuda().numpy()
                true_labels = labels.cuda().numpy()

                predictions.extend(predicted_labels)
                targets.extend(true_labels)

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)

        return avg_loss, accuracy, precision, recall, f1
    
    def evaluate(self, eval_data, eval_labels):
        eval_labels = eval_labels.to('cuda')
        self.eval()
        with torch.no_grad():
            predictions = self(eval_data)
            predicted_labels = predictions.round().long()
            correct = (predicted_labels == eval_labels).sum().item()
            accuracy = correct / len(eval_labels)
        eval_labels = eval_labels.cuda()
        eval_data = eval_data.cuda()
        return accuracy

   # def loadModel(self, path):
        #self.
        

    
"""

def trainModel(self, train_data, train_labels, validation_data, validation_labels, num_epochs, batch_size):
        nz = 0#[ if x.item() == [0, 1] then 1 else 0 for x in train_labels ]# (train_labels == 0).sum().item()
        po = 0
        for x in train_labels:
            x = list(x.numpy())
            if x == [0, 1]:
                nz += 1
            else:
                po += 1
        #imbalance_ratio
        #imbalance_ratio = nz / po
        nz = nz / len(train_labels)
        po = po / len(train_labels)
        #imbalance_ratio = imbalance_ratio if imbalance_ratio > 1 else 1 / imbalance_ratio
        weight = torch.tensor([nz, po])
        #pos_weight = torch.tensor([imbalance_ratio]).to(self.device)
        self.criterion = nn.BCELoss(weight=weight).to(self.device)
        self.train()
        best_loss = -float('inf')
        patience = 16
        best_epoch = 0
        num_batches = (len(train_data) - 1) // batch_size + 1
        num_batches = (len(train_data) - 1) // batch_size + 1
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                batch_inputs = train_data[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                batch_labels = batch_labels.squeeze().to(self.device)

                self.optimizer.zero_grad()
                output = self.forward(batch_inputs)
                o = output#.squeeze(2)
                loss = self.criterion(o, batch_labels.float())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            lo = o[0]
            lo = list(lo.detach().numpy())

            # Perform validation and check for early stopping
            accuracy = -1
            if validation_data is not None and validation_labels is not None:
                accuracy = self.evaluate(validation_data, validation_labels)
                if accuracy > best_loss:
                    best_loss = accuracy
                    best_epoch = epoch
                elif epoch - best_epoch >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, acc: {accuracy:.4f}, Last: {lo[0]:.4f}")

    def evaluate(self, eval_data, eval_labels):
        eval_labels = eval_labels.to(self.device)
        self.eval()
        with torch.no_grad():
            predictions = self.forward(eval_data)
            maxs = torch.argmax(predictions, dim=1)
            maxl = torch.argmax(eval_labels, dim=1)
            correct = (maxs == maxl).sum().item()
            accuracy = correct / len(maxl)
        return accuracy

"""




"""
    def trainModel(self, train_data, train_labels, validation_data, validation_labels, num_epochs, batch_size):
        nz = (train_labels == 0).sum().item()
        po = (train_labels == 1).sum().item()
        #imbalance_ratio
        imbalance_ratio = nz / po
        #imbalance_ratio = imbalance_ratio if imbalance_ratio > 1 else 1 / imbalance_ratio
        weight = torch.tensor([imbalance_ratio])
        patience = 16
        pos_weight = torch.tensor([imbalance_ratio]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        self.train()
        best_loss = float('inf')
        best_epoch = 0
        num_batches = (len(train_data) - 1) // batch_size + 1
        num_batches = (len(train_data) - 1) // batch_size + 1
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                batch_inputs = train_data[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                batch_labels = batch_labels.squeeze().to(self.device)

                self.optimizer.zero_grad()
                output = self.forward(batch_inputs)
                o = output.squeeze()
                loss = self.criterion(o, batch_labels.float())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            lo = list(o)[0].item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Last: {lo:.4f}")
            # Perform validation and check for early stopping
            if validation_data is not None and validation_labels is not None:
                val_loss = self.evaluate(validation_data, validation_labels)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                elif epoch - best_epoch >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def evaluate(self, eval_data, eval_labels):
        eval_labels = eval_labels.to(self.device)
        self.eval()
        with torch.no_grad():
            predictions = self.forward(eval_data)
            predicted_labels = predictions.round().squeeze().long()
            correct = (predicted_labels == eval_labels).sum().item()
            accuracy = correct / len(eval_labels)
        return accuracy
"""