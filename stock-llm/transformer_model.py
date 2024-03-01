import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, input_dim)
        output = self.decoder(output)
        return output

def train_transformer_model(data, input_dim, output_dim, nhead, num_layers, batch_size, num_epochs, learning_rate):
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, nhead=nhead, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            src = batch.float()
            tgt = batch.float()  # Example: Target is same as source for simplicity
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

    return model