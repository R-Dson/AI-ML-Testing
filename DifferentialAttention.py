
# https://arxiv.org/abs/2410.05258

import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentialAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, lambda_init=0.8, layer_index=0, num_layers=12):
        super(DifferentialAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Linear layers for queries, keys, and values (for both softmaxes)
        self.qkv_proj = nn.Linear(d_model, d_model * 2)  # Produces Q1, Q2, K1, K2
        self.value_proj = nn.Linear(d_model, d_model * 2 * n_heads)  # For values (V)
        
        # Final projection after attention
        self.out_proj = nn.Linear(d_model * 2 * n_heads, d_model)
        
        # Lambda reparameterization: learnable vectors for lambda
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))  # Vector for lambda_q1
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))  # Vector for lambda_k1
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))  # Vector for lambda_q2
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))  # Vector for lambda_k2
        
        # Lambda initialization and layer-specific decay
        self.layer_index = layer_index
        self.lambda_init = lambda_init
        self.num_layers = num_layers

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def get_lambda(self):
        # Not sure about this part yet.
        lambda_q1k1 = torch.sum(self.lambda_q1 * self.lambda_k1)
        lambda_q2k2 = torch.sum(self.lambda_q2 * self.lambda_k2)
        
        lambda_ = torch.exp(lambda_q1k1) - torch.exp(lambda_q2k2)

        lambda_decay = self.lambda_init - 0.6 * torch.exp(torch.tensor([-0.3 *self.layer_index]).cuda())
        
        lambda_final = lambda_ + lambda_decay

        return lambda_final

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, d_model = x.size()

        qkv = self.qkv_proj(x).chunk(4, dim=-1)  # Split into Q1, Q2, K1, K2
        Q1, Q2, K1, K2 = qkv[0], qkv[1], qkv[2], qkv[3]

        # Project input to get values V
        V = self.value_proj(x)
        
        # Compute attention scores for both sets of queries and keys
        attn1 = (Q1 @ K1.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn2 = (Q2 @ K2.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask (if provided)
        if attn_mask is not None:
            extended_attn_mask = attn_mask.unsqueeze(1).expand(-1, 512, -1)
            attn1 = attn1.masked_fill(extended_attn_mask == 0, float('-inf'))
            attn2 = attn2.masked_fill(extended_attn_mask == 0, float('-inf'))

        # Apply softmax to the attention scores
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Get the dynamically updated lambda for this layer
        lambda_final = self.get_lambda()

        # Differential attention: Subtract the two softmax outputs
        differential_attn = attn1 - lambda_final * attn2
        
        # Multiply by values V to get the final output
        output = differential_attn @ V
        output = self.dropout(output)
        output = self.out_proj(output)

        return output


class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, lambda_init=0.5):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        # Define differential attention for multiple heads
        self.attention_heads = nn.ModuleList([
            DifferentialAttention(d_model, n_heads, dropout, lambda_init)
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model * n_heads, d_model)
    
    def forward(self, x, attn_mask=None):
        # Apply differential attention independently to each head
        attn_outputs = [head(x, attn_mask) for head in self.attention_heads]

        # Concatenate the outputs from all heads
        concatenated_output = torch.cat(attn_outputs, dim=-1)
        
        # Project the concatenated output back to d_model size
        output = self.out_proj(concatenated_output)

        return output

class DIFFTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, lambda_init=0.5):
        super(DIFFTransformerLayer, self).__init__()
        self.attention = MultiHeadDifferentialAttention(d_model, n_heads, dropout, lambda_init)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network (SwiGLU activation)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        ).to('cuda')

    def forward(self, x, attn_mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, attn_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class DIFFTransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, vocab_size, max_seq_len=512, dropout=0.1, lambda_init=0.5):
        super(DIFFTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DIFFTransformerLayer(d_model, n_heads, dropout, lambda_init)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)


    def create_positional_encoding(self, max_seq_len, d_model):
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x)

        x = x + self.positional_encoding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)
        
        x = self.norm(x)
        logits = self.output_proj(x)

        return logits
