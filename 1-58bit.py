import math
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

tokenizer = PreTrainedTokenizerFast.from_pretrained('Xenova/Meta-Llama-3.1-Tokenizer')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(64, max_len, d_model).cpu()
        pe[:,:, 0::2] = torch.sin(position * div_term)
        pe[:,:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.transpose(0, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0),:x.size(1),:].cuda()
        return x

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))  # Change here
        self.epsilon = 1e-8
    
    def get_gamma(self, W):
        n = W.shape[0]
        m = W.shape[1]
        gamma = (1 / (n * m)) * torch.sum(torch.abs(W))
        return gamma
    
    def round_clip(self, x, a, b):
        return torch.clamp(torch.round(x), a, b)
    
    def clip_weights(self, W, gamma):
        W_s = W / (gamma + self.epsilon)
        W_hat = self.round_clip(W_s, -1, 1)
        return W_hat

    def forward(self, x):
        gamma = self.get_gamma(self.weight)
        W_hat = self.clip_weights(self.weight, gamma)
        y = torch.matmul(x, W_hat)
        return y

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, drop_prob=0.1):
        super().__init__()
        self.masked_multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(drop_prob)
        )

    def forward(self, x, attn_mask=None):
        if attn_mask is not None:
            batch_size = x.size(0)
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
            attn_mask = attn_mask.bool()
            attn_mask = attn_mask[:, :, 0].cuda()
            attn_output = self.masked_multihead_attn(x, x, x, attn_mask=attn_mask)[0]
        else:
            attn_output = self.masked_multihead_attn(x, x, x)[0]
        out = self.feed_forward(attn_output)
        return out + x

    
class BitNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, MAX_SEQ_LENGTH)
        self.decoder = nn.ModuleList([DecoderBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.bitlinear = BitLinear(embedding_dim, vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, x, attention_mask):
        x = x.to(self.device)
        x = self.embedding(x)
        x = self.pos_encoding(x)

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
                    try:
                        r = generate_text(model, tokenizer, 'The red ')
                        print(f'{r}\n')
                    except:
                        print('Error generating text')
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
