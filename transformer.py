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

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))  # bias is a learnable parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.alpha.to('cuda') * (x - mean) / (std + self.eps) + self.bias.to('cuda')


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        return torch.sigmoid(self.linear_2(self.dropout(torch.relu(self.linear_1(x)))))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = x.to('cuda')
        return self.embedding.to('cuda')(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe.to('cuda')[:, :x.shape[1], :]).requires_grad_(False)
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
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        return torch.matmul(attention_weights, value)

    def forward(self, x, mask=None):
        residual = x
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)
        batch_size = x.size(0)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        x = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(self.dropout(x)) + residual


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, d_ff: int, h: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ResidualConnection(dropout) for _ in range(num_layers)])
        self.layer_norm = LayerNormalization()
        self.num_layers = num_layers

        for i in range(num_layers):
            self.layers[i].sublayer = MultiHeadAttentionBlock(d_model, h, dropout)
            self.layers[i].feed_forward = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        for i in range(self.num_layers):
            if mask is not None:
                x = self.layers[i](x, lambda x: x.sublayer(x, mask))
                x = self.layers[i](x, lambda x: x.feed_forward(x))
        return self.layer_norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, h: int, num_layers: int, dropout: float, maxlen: int) -> None:
        super().__init__()
        self.input_embeddings = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_len=maxlen, dropout=dropout)
        self.encoder = TransformerEncoder(d_model, d_ff, h, num_layers, dropout)
        self.output_layer = nn.Linear(d_model, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-9)

    def forward(self, x, mask=None):
        x = self.input_embeddings(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        x = self.output_layer.to('cuda')(x[:, 0, :])  # consider only the first token's output
        return torch.sigmoid(x)
    
    def trainModel(self, train_data, train_labels, val_loader, num_epochs, batch_size, lr=0.001):
        nz = 0#[ if x.item() == [0, 1] then 1 else 0 for x in train_labels ]# (train_labels == 0).sum().item()
        po = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """for x in train_labels:
            x = list(x.numpy())
            if x == [0, 1]:
                nz += 1
            else:
                po += 1"""
        #imbalance_ratio
        #imbalance_ratio = nz / po
        nz = (train_labels == 0).sum().item()
        po = (train_labels == 1).sum().item()
        nz = nz / len(train_labels)
        po = po / len(train_labels)
        #imbalance_ratio = imbalance_ratio if imbalance_ratio > 1 else 1 / imbalance_ratio
        #weight = torch.tensor([nz/po])
        #pos_weight = torch.tensor([imbalance_ratio]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1/po]))
        self.train()
        best_loss = -float('inf')
        patience = 16
        best_epoch = 0
        num_batches = (len(train_data) - 1) // batch_size + 1
        num_batches = (len(train_data) - 1) // batch_size + 1

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                batch_inputs = train_data[start_idx:end_idx].to(device)
                batch_labels = train_labels[start_idx:end_idx]
                batch_labels = batch_labels.to(device)

                self.criterion.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    output = self(batch_inputs)
                    loss = self.criterion.cuda()(output, batch_labels.float())
                scaler.scale(loss).backward()
                self.cuda()
                scaler.step(self.optimizer)
                scaler.update()

                #output = self.forward(batch_inputs)
                #o = output
                #loss = self.criterion.to('cuda')(output, batch_labels.float())
                #loss.backward()
                #self.optimizer.step()
                #ans = output.round

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            lo = output.mean().cpu()
            #lo = list(lo.detach().numpy())

            # Perform validation and check for early stopping
            accuracy = -1
            lo = lo.item()
            if val_loader is not None:
                accuracy = self.evaluate3(val_loader)
                if accuracy > best_loss:
                    best_loss = accuracy
                    best_epoch = epoch
                elif epoch - best_epoch >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, acc: {accuracy:.4f}, avg val: {lo:.9f}")

    def count_parameters(self):
         return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def trainModel(self, train_loader, validation_loader, num_epochs, batch_size, lr=0.001, po = 1, nz = 1):
        #nz = 0#[ if x.item() == [0, 1] then 1 else 0 for x in train_labels ]# (train_labels == 0).sum().item()
        #po = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', device)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-9)
        print('optimizer')
        """for x in train_labels:
            x = list(x.numpy())
            if x == [0, 1]:
                nz += 1
            else:
                po += 1"""
        #imbalance_ratio
        #imbalance_ratio = nz / po
        #nz = (train_labels == 0).sum().item()
        #po = (train_labels == 1).sum().item()
        #nz = nz / len(train_labels)
        #po = po / len(train_labels)
        #imbalance_ratio = imbalance_ratio if imbalance_ratio > 1 else 1 / imbalance_ratio
        #weight = torch.tensor([nz/po])
        #pos_weight = torch.tensor([imbalance_ratio]).to(self.device)
        if hasattr(self, 'criterion') is False:
            print('criterion')
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([po/nz]))
        self.train()
        best_loss = -float('inf')
        patience = 12
        best_epoch = 0
        trl = len(train_loader)
        num_batches = (trl - 1) // batch_size + 1
        print('num_batches', num_batches)
        self.cuda()
        #num_batches = (len(train_data) - 1) // batch_size + 1

        scaler = torch.cuda.amp.GradScaler()
        print('starting training loop')
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            bi = 0
            for batch_inputs, batch_labels in train_loader:
                batch_labels = batch_labels.type(torch.float16)
                bi += 1
            #for batch_idx in range(num_batches):
                #start_idx = batch_idx * batch_size
                #end_idx = (batch_idx + 1) * batch_size
                #batch_inputs = train_data[start_idx:end_idx].to(device)
                #batch_labels = train_labels[start_idx:end_idx]
                #batch_labels = batch_labels.to(device)

                with torch.cuda.amp.autocast():
                    output = self(batch_inputs.to('cuda', non_blocking=True))
                    #self.criterion = self.criterion#.cuda()
                    loss = self.criterion(output, batch_labels.to('cuda', non_blocking=True))
                    #self.criterion = self.criterion.cpu()
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    

                    scaler.step(self.optimizer)
                    scaler.update()

                #self.cpu()
                if bi % 1000 == 0:
                    print(f"Epoch {bi}/{trl}")
                
                #output = self.forward(batch_inputs)
                #o = output
                #loss = self.criterion.to('cuda')(output, batch_labels.float())
                #loss.backward()
                #self.optimizer.step()
                #ans = output.round

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            lo = output.mean().cpu()
            #lo = list(lo.detach().numpy())


            loss, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
            if validation_loader is not None:
                loss, accuracy, precision, recall, f1 = self.evaluate3(validation_loader)
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch

            lo = output.mean().cpu().item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, loss: {loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            """
            # Perform validation and check for early stopping
            loss = -1
            lo = lo.item()
            if validation_loader is not None:
                loss, accuracy, precision, recall, f1 = self.evaluate3(validation_loader)
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                #elif epoch - best_epoch >= patience:
                    #print(f"Early stopping at epoch {epoch+1}")
                    #break
                #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, loss: {loss:.4f}, avg val: {lo:.9f}")
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            """
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

                predicted_labels = torch.round(torch.sigmoid(outputs)).cpu().numpy()
                true_labels = labels.cpu().numpy()

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
        eval_labels = eval_labels.cpu()
        eval_data = eval_data.cpu()
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
