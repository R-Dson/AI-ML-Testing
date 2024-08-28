import math
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

tokenizer = PreTrainedTokenizerFast.from_pretrained('Xenova/Meta-Llama-3.1-Tokenizer')

class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, epsilon=1e-8):
        ctx.save_for_backward(input, gamma)
        ctx.epsilon = epsilon
        W_s = input / (gamma + epsilon)
        W_hat = torch.clamp(torch.round(W_s), -1, 1)
        return W_hat

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma = ctx.saved_tensors
        epsilon = ctx.epsilon
        grad_input = grad_output / (gamma + epsilon)
        return grad_input, None, None

quantize = QuantizeFunction.apply

class ActivationQuantization(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x):
        max_val = torch.max(torch.abs(x))
        scale = (2 ** (self.num_bits - 1) - 1) / max_val
        return torch.round(x * scale) / scale

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        x_norm = x / (rms + self.eps)
        return self.weight * x_norm

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = BitLinear(in_features, hidden_features)
        self.w2 = BitLinear(in_features, hidden_features)
        self.w3 = BitLinear(hidden_features, in_features)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = torch.sigmoid(x1) * x2
        return self.w3(hidden)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

        # Ensure the cosine and sine have the correct dimension
        cos = torch.cat([self.cos_cached, self.cos_cached], dim=-1)
        sin = torch.cat([self.sin_cached, self.sin_cached], dim=-1)

        return cos[:, :, :seq_len, :], sin[:, :, :seq_len, :]
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q and k shapes: [seq_len, num_heads, head_dim]
    # cos and sin shapes: [1, 1, seq_len, head_dim]

    # Reshape cos and sin to [seq_len, num_heads, head_dim] by unsqueezing and broadcasting
    cos = cos.squeeze().unsqueeze(1) # Add dimensions for broadcasting
    sin = sin.squeeze().unsqueeze(1)
    
    if q.shape[0] == 1:
        cos = cos.transpose(1, 0).unsqueeze(1)
        sin = sin.transpose(1, 0).unsqueeze(1)

    # Broadcast to match q and k shapes
    cos = cos.expand_as(q)
    sin = sin.expand_as(q)

    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))  # Changed order
        self.epsilon = 1e-8

    def get_gamma(self, W):
        return torch.mean(torch.abs(W))

    def forward(self, x):
        gamma = self.get_gamma(self.weight)
        W_hat = quantize(self.weight, gamma, self.epsilon)
        return torch.matmul(x, W_hat.t())  # Transposed here

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, drop_prob=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.feed_forward = SwiGLU(embedding_dim, hidden_dim)
        self.norm1 = RMSNorm(embedding_dim)
        self.norm2 = RMSNorm(embedding_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.act_quant = ActivationQuantization(num_bits=8)
        self.rotary_emb = RotaryEmbedding(embedding_dim // num_heads)

    def forward(self, x, attn_mask=None):
        # Self-attention with rotary embeddings
        q, k, v = x, x, x
        seq_len = q.size(0)
        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if attn_mask is not None:
            ext_size = x.size(0)
            attn_mask = attn_mask.unsqueeze(0).expand(ext_size, -1, -1)
            attn_mask = attn_mask.bool()
            attn_mask = attn_mask[:, :, 0].cuda()
        attn_output, _ = self.self_attn(q, k, v, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.act_quant(x)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        x = self.act_quant(x)
        
        return x

class BitNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.ModuleList([DecoderBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.bitlinear = BitLinear(embedding_dim, vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, attention_mask):
        x = x.to(self.device)
        x = self.embedding(x)

        for decoder_block in self.decoder:
            x = decoder_block(x, attention_mask)

        out = self.bitlinear(x)
        return out


def validate(model, val_dataloader, loss_fn):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            outputs = model(input_ids, attention_mask)
            outputs = outputs.view(-1, vocab_size)

            loss = loss_fn(outputs, input_ids.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs):
    max_grad_norm = 1.0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_dataloader):
            input_ids = torch.stack(batch['input_ids']).to(model.device)  # Send to GPU here
            attention_mask = torch.stack(batch['attention_mask']).to(model.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                outputs = outputs.view(-1, vocab_size)
                loss = loss_fn(outputs, input_ids.view(-1))
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if i != 0:
                avg_loss = total_loss / i
                if (i + 1) % 5 == 0:
                    r = generate_text(model, tokenizer, 'The red ')
                    print(f'{r}\n')
                    #try:
                    #except:
                    #    print('Error generating text')
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        scheduler.step()

        validate(model, val_dataloader, loss_fn)

    torch.save(model.state_dict(), 'bitnet_checkpoint.pth')  

MAX_SEQ_LENGTH = 1024

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=None)
            
            logits = outputs[:, -1, :] / temperature
            
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits = torch.zeros_like(logits).scatter_(-1, indices, values)
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            
            if next_token_id == tokenizer.eos_token_id:
                break

    # Decode the generated sequence of tokens into text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return generated_text

def tokenize_function(examples):
    return tokenizer(examples['Response'], truncation=True, max_length=MAX_SEQ_LENGTH, padding='max_length')

if __name__ == "__main__":
    
    tokenizer.pad_token = tokenizer.eos_token
    MAX_SEQ_LENGTH = tokenizer.model_max_length // 64
    vocab_size = tokenizer.vocab_size+10
    batch_size = 4
    embedding_dim = 256
    num_layers = 4
    num_heads = 2
    hidden_dim = 512
    model = BitNet(vocab_size, embedding_dim, num_layers, num_heads, hidden_dim).cuda()

    dataset = load_dataset('Amod/mental_health_counseling_conversations', split='train')

    def tokenize_function(examples):
        return tokenizer(examples['Response'], truncation=True, max_length=MAX_SEQ_LENGTH, padding='max_length')

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['Response'])
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=3)
