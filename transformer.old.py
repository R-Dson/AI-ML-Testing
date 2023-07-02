#import tensorflow as tf
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.layers import Embedding, Flatten
from keras.models import Model
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
from torchinfo import summary
import tiktoken
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import numpy as np
import os
from tqdm import tqdm
"""
class TransformerModel:
    def __init__(self, max_seq_len, vocab_size, num_heads, num_layers, d_model, dff, dropout_rate):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Input layer
        self.inputs = Input(shape=(self.max_seq_len,), dtype='int32')
        
        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.d_model)
        embedded = self.embedding(self.inputs)
        
        # Positional encoding layer
        self.pos_encoding = self._get_positional_encoding()
        encoded = embedded + self.pos_encoding[:, :self.max_seq_len, :]
        
        # Dropout layer
        encoded = Dropout(self.dropout_rate)(encoded)
        
        # Transformer layers
        for i in range(self.num_layers):
            encoded = self._add_transformer_layer(encoded, i)
        
        # Output layer
        self.outputs = Dense(1, activation='sigmoid')(encoded)
        
        # Create model
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
    
    def _get_positional_encoding(self):
        pos = tf.range(self.max_seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / self.d_model)
        angle_rads = pos * angle_rates

        # apply sin to even indices in the array; 2i
        sin_even = tf.math.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cos_odd = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sin_even, cos_odd], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _add_transformer_layer(self, inputs, i):
        # Multi-head self-attention layer
        attn_output = MultiHeadAttention(num_heads=self.num_heads,
                                          key_dim=self.d_model)(inputs, inputs)
        attn_output = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        
        # Feedforward layer
        ffn_output = Dense(self.dff, activation='relu')(attn_output)
        ffn_output = Dense(self.d_model)(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(attn_output + ffn_output)
        
        return ffn_output
    
    def compile_model(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, X_train, y_train, epochs, X_val=None, y_val=None, batch_size=32):
        self.history = self.model.fit(X_train, y_train,
                                       epochs=epochs, batch_size=batch_size)
    
    def build(self):
        self.model.build(input_shape=(None, self.max_seq_len))
        self.model.summary()

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save(path)

"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.query = nn.Linear(self.key_dim, self.key_dim * self.num_heads, bias=False)
        self.key = nn.Linear(self.key_dim, self.key_dim * self.num_heads, bias=False)
        self.value = nn.Linear(self.key_dim, self.key_dim * self.num_heads, bias=False)
        self.dense = nn.Linear(self.key_dim * self.num_heads, self.key_dim)

    
    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        query = self.query(inputs).view(batch_size, seq_len, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        key = self.key(inputs).view(batch_size, seq_len, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        value = self.value(inputs).view(batch_size, seq_len, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        score = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32))
        attention_weights = F.softmax(score, dim=-1)
        output = torch.matmul(attention_weights, value).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.dense(output)
        return output



class TransformerLayer(nn.Module):
    def __init__(self, num_heads, key_dim, dff, dropout_rate):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, key_dim)
        self.ffn = nn.Sequential(
            nn.Linear(key_dim, dff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, key_dim),
            nn.Dropout(dropout_rate)
        )
        self.layernorm1 = nn.LayerNorm(key_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(key_dim, eps=1e-6)
        
    def forward(self, inputs):
        attn_output = self.mha(inputs)
        attn_output = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(attn_output)
        ffn_output = self.layernorm2(attn_output + ffn_output)
        return ffn_output

class Model(nn.Module):
    def __init__(self, max_seq_len, vocab_size, num_heads, num_layers, d_model, dff, dropout_rate):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.hasPrint = False
        self.to('cuda')
        self.lossfn = nn.CrossEntropyLoss()
        self.lossfn.to('cuda')
        self.optimizer = None
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Positional encoding layer
        self.pos_encoding = self._get_positional_encoding()
        self.pos_encoding.to('cuda')
        self.pos_encoding.cuda()
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # Transformer layers
        #self.transformer_layers = nn.ModuleList(
        #    [TransformerLayer(num_heads, d_model, dff, dropout_rate) for _ in range(num_layers)]
        #)
        self.transformer_layers = nn.ModuleList([
            self._add_transformer_layer() for _ in range(self.num_layers)
        ])

        # Output layer
        #self.output_layer = nn.Linear(d_model, 1)
        self.dense = nn.Linear(self.max_seq_len * self.d_model, 1)
        self.activation = nn.Sigmoid()
        self.expected_moved_cuda_tensor = torch.nn.Parameter(self.pos_encoding)
        

    def forward(self, inputs):
        embedded = self.embedding(inputs.int())
        device = torch.device('cuda:0')
        self.pos_encoding.cuda()
        self.pos_encoding.to(device)
        t = self.pos_encoding
        t.to(device)
        t.cuda()
        t.to('cuda')
        encoded = embedded + t[:, :self.max_seq_len, :]
        encoded = F.dropout(encoded, p=self.dropout_rate, training=self.training)
        
        for layer in self.transformer_layers:
            encoded = layer(encoded)
        
        flattened = encoded.view(encoded.size(0), -1)
        output = self.dense(flattened)
        output = self.activation(output)
        return output

    
    def _get_positional_encoding(self):
        pos = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        pos.cuda()
        i = torch.arange(self.d_model, dtype=torch.float32).unsqueeze(0)
        i.cuda()
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / self.d_model)
        angle_rads = pos * angle_rates

        # apply sin to even indices in the array; 2i
        sin_even = torch.sin(angle_rads[:, 0::2])
        sin_even.cuda()
        # apply cos to odd indices in the array; 2i+1
        cos_odd = torch.cos(angle_rads[:, 1::2])
        cos_odd.cuda()

        pos_encoding = torch.cat([sin_even, cos_odd], dim=-1)
        pos_encoding = pos_encoding.unsqueeze(0)
        
        return pos_encoding.cuda()
        
    def _add_transformer_layer(self):
        layer = nn.Sequential(
            MultiHeadAttention(self.num_heads, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.dff),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dff, self.d_model),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.d_model)
        )
        return layer      
    
    def trainModel(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        #self.history = self.model.fit(X_train, y_train,
        #                               epochs=epochs, batch_size=batch_size)
        self.train()
        lr=0.01
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=batch_size)
        
        for epoch in range(epochs):
            i = 0
            correct = 0
            total = 0
            totalloss = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to('cuda')
                y_batch = y_batch.to('cuda')
                
                self.optimizer.zero_grad()

                ypred = self(x_batch)
                a = ypred.squeeze().round()
                b = y_batch.float().squeeze()

                loss = self.lossfn(a, b)#self.lossfn(a, b)

                loss.backward()
                self.optimizer.step()

                i = i + 1
                totalloss += loss.item()
                correct += (a == b).int().sum()
                total += len(x_batch)
                if i % 100 == 0:
                    pass
            accuracy = 100 * correct.item() / total
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch+1, epochs, i+1, len(loader), totalloss/i, accuracy))
                    
                    #print("Accuracy = {}%".format(accuracy))
        """
"""
        self.eval()
        loader = DataLoader(list(zip(X_val, y_val)), shuffle=True, batch_size=batch_size)
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in loader:
                labels.to('cuda')
                self.expected_moved_cuda_tensors = torch.nn.Parameter(labels.float())
                outputs = self(inputs.to('cuda'))
                a = outputs.round().squeeze().int()
                #_, predicted = torch.max(a, 1)
                total += labels.size(0)
                labels = labels.flatten().int()
                a = a.to('cpu')
                labels = labels.to('cpu')
                #ai = a.to('cpu').numpy()
                #print(a.item())
                #li = labels.item()
                test = (a == labels)
                correct += (a == labels).sum().item()
            print('Accuracy of the network on the test: %d %%' % (100 * correct / total))
        self.train()"""
"""
    def build(self):
        #self.model.build(input_shape=(None, self.max_seq_len))
        self.summary()

    def printsummary(self, id):
        f = id[0]
        if self.hasPrint == False:
            self.hasPrint = True
            summary(self, input_size=f.shape)

    def evaluate(self, X_test, y_test):
        self.eval()
        loader = DataLoader(list(zip(X_test, y_test)), shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                labels.to('cuda')
                self.expected_moved_cuda_tensors = torch.nn.Parameter(labels.float())
                outputs = self(inputs.to('cuda'))
                a = outputs.round().squeeze().int()
                total += labels.size(0)
                labels = labels.flatten().int()
                a = a.to('cpu')
                labels = labels.to('cpu')
                correct += (a == labels).sum().item()
            print('Accuracy of the network on the test: %d %%' % (100 * correct / total))
        self.train()
        return (100 * correct / total)
        #return self.eval(X_test, y_test)
    def predict(self, X):
        self.eval()
        return self(X)
    #def predict(self, X):
        #return self.predict(X)

    def save_model(self, path):
        torch.save(self, path)
    """


import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow as tf
from tensorflow import keras
from keras import layers

class TransformerEncoderG(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(TransformerEncoderG, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.positional_encoding = self.positional_encoding(maximum_position_encoding, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.enc_layers = [self.encoder_layer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(d_model, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid')

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sinusoidal function to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cosine function to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def encoder_layer(self, d_model, num_heads, dff, dropout_rate):
        inputs = keras.Input(shape=(None, d_model))
        attention = layers.MultiHeadAttention(num_heads, d_model)
        attention_output = attention(inputs, inputs)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        ffn = self.point_wise_feed_forward_network(d_model, dff)
        ffn_output = ffn(attention_output)
        ffn_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        return keras.Model(inputs=inputs, outputs=ffn_output)

    def point_wise_feed_forward_network(self, d_model, dff):
        return keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, inputs, training=True):
        seq_len = tf.shape(inputs)[1]
        # Add embedding and position encoding.
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)

        x = self.final_layer(x)
        flattened_output = self.flatten(x)

        # Final dense layer for binary classification
        logits = self.dense_output(flattened_output)

        return logits

        
        #return x
    def trainer(self, train_dataset, loss_function, optimizer, num_epochs):
        self.compile(optimizer=optimizer, loss=loss_function)
        self.fit(train_dataset, epochs=num_epochs)

    def train(self, train_dataset, loss_function, optimizer, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for inputs, labels in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self(inputs, training=True)
                    predictions = tf.reduce_mean(predictions, axis=1)  # Average predictions across the second dimension
                    labels = tf.reshape(labels, (-1,))  # Reshape labels to match logits shape
                    loss = loss_function(labels, predictions)
                    
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                epoch_loss += loss.numpy()
                num_batches += 1

            average_loss = epoch_loss / num_batches

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")


""""""""""""""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

    
class TransformerClassifier(nn.Module):
    def __init__(self, num_tokens, emb_dim, num_heads, hidden_dim, num_layers, max_seq_length):
        torch.manual_seed(42)
        super(TransformerClassifier, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        self.embedding = nn.Embedding(num_tokens, emb_dim)
        self.positional_encoding = self._create_positional_encoding(emb_dim, max_seq_length)
        self.encoder_layer = nn.TransformerEncoderLayer(emb_dim, num_heads, hidden_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, input_tokens):
        if input_tokens.device != self.device:
            input_tokens = input_tokens.to(self.device)
        embedded = self.embedding.to(self.device)(input_tokens) + self.positional_encoding.to(self.device)
        encoded = self.encoder.to(self.device)(embedded)
        pooled = self.pooling.to(self.device)(encoded.permute(0, 2, 1)).squeeze()
        output = self.fc.to(self.device)(pooled)
        output = self.sigmoid.to(self.device)(output)
        return output

    def _create_positional_encoding(self, emb_dim, max_seq_length):
        encoding = torch.zeros(max_seq_length, emb_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)
    
    def trainModel(self, X, y, num_epochs, batch_size, wandb=None, lr=0.001, weight_decay=0.001):
        z = (y == 0).sum()  # Number of negative instances
        e = (y == 1).sum()  # Number of positive instances
        pos_weight = e/z  # Calculate pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float)).to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        patience = 6

        validation_ratio = 0.1

        valdata = X[int(len(X)*(1-validation_ratio)):]
        valans = y[int(len(X)*(1-validation_ratio)):]

        X = X[:int(len(X)*(1-validation_ratio))]
        y = y[:int(len(X)*(1-validation_ratio))]

        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=batch_size)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
        dl = len(dataloader)

        best_eval = 0
        for epoch in range(num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            loop = tqdm(dataloader, total=dl)
            batch_num = 0
            for batch_inputs, batch_labels in loop:
                optimizer.zero_grad()
                batch_labels = batch_labels.float().to(self.device)
                j = 0
                for x in batch_inputs:
                    outputs = self(x.to(self.device)).squeeze()
                    gt = batch_labels[j]
                    loss = criterion(outputs, gt)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    predicted_labels = (outputs >= 0.5).long()
                    #correct = (predicted_labels == batch_labels.long()).sum().item()
                    correct = (predicted_labels.item() == int(gt.item()))#.sum().item()
                    j += 1
                    if correct:
                        total_correct += 1
                    #total_correct += correct
                
                #outputs = self(batch_inputs.to(self.device))
                #outputs = outputs.round()
                #outputs = (outputs >= 0.5).float()
                
                
                
                
                total_samples += batch_labels.size(0)
                accuracy = total_correct / total_samples * 100
                batch_num += 1

                #if total_samples % 50 == 0:
                learnrate = scheduler.get_last_lr()[0]
                loop.set_description(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_num}/{dl}], lr [{learnrate:.6f}]")
                loop.set_postfix(loss=loss.item(), acc=accuracy)
            #if batch_num % 100 == 0:
                    #wandb.log({"loss": loss, "accuracy": accuracy})

            eval = self.eval(valdata, valans)
            scheduler.step()

            if eval > best_eval:
                best_eval = eval
                patience_count = 0 
            else:
                patience_count += 1

            if patience_count >= patience:
                print("Early stopping triggered. No improvement in validation accuracy.")
                break

    def pred(self, x):
        test_outputs = self(x)
        test_outputs = (test_outputs >= 0.5).squeeze().long().item()
        return test_outputs

    def eval(self, X, y):
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=1)
        total_correct = 0
        total_samples = 0
        loop = tqdm(dataloader)
        for batch_inputs, batch_labels in loop:
            batch_labels = batch_labels.float().to(self.device)
            outputs = self(batch_inputs.to(self.device))
            predicted_labels = (outputs >= 0.5).long() # ################################## CHANGE THRESHOLD LATER
            correct = (predicted_labels == batch_labels.long()).sum().item()
            total_correct += correct
            total_samples += batch_labels.size(0)
        accuracy = total_correct / total_samples * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    def save(self, path):
        torch.save(self.state_dict(), path)
    once = False
    def printModel(self):
        if self.once == True:
            return
        print(self)
        self.once = True

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.to('cuda')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(embedding_dim, 1)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, x):
        x = x.to(self.device)
        embedded = self.embedding.to(self.device)(x)  # shape: (batch_size, seq_len, embedding_dim)
        encoded = self.positional_encoding.to(self.device)(embedded)
        
        for attention_block in self.attention_blocks:
            encoded = attention_block.to(self.device)(encoded)
            
        logits = self.fc.to(self.device)(encoded[:, 0, :])  # Taking the representation of the first token
        output = torch.sigmoid(logits)
        
        return output
    
    def trainModel(self, X, y, num_epochs, batch_size, wandb=None, lr=0.001, weight_decay=0.001):
        z = (y == 0).sum()  # Number of negative instances
        e = (y == 1).sum()  # Number of positive instances
        pos_weight = z/e  # Calculate pos_weight
        #if z > 0 and e > 0:
        #    pos_weight = z/e
        #else:
        #    pos_weight = 1.0  # Set default value to 1 if there are no positive or negative instances
        lz = z / (z + e)
        ze = e/z
        criterion = nn.BCELoss(weight=torch.tensor([ze])).to(self.device)
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float)).to(self.device)
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float)).to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        patience_count = 0
        patience = 6

        validation_ratio = 0.1

        valdata = X[int(len(X)*(1-validation_ratio)):]
        valans = y[int(len(X)*(1-validation_ratio)):]

        X = X[:int(len(X)*(1-validation_ratio))]
        y = y[:int(len(X)*(1-validation_ratio))]

        if len(X) < 1: return

        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=batch_size)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
        dl = len(dataloader)

        best_eval = 0
        for epoch in range(num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            loop = tqdm(dataloader, total=dl)
            batch_num = 0
            avgacc = 0
            for batch_inputs, batch_labels in loop:
                optimizer.zero_grad()
                batch_labels = batch_labels.float().to(self.device)
                outputs = self(batch_inputs.to(self.device))
                #outputs = outputs.round()
                #outputs = (outputs >= 0.5).float()
                outputs = outputs.squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                predicted_labels = (outputs >= lz).long()
                correct = (predicted_labels == batch_labels.long()).sum().item()
                total_correct += correct
                total_samples += batch_labels.size(0)
                accuracy = total_correct / total_samples * 100
                batch_num += 1

                avgacc += accuracy

                if total_samples % 5 == 0:
                    #learnrate = scheduler.get_last_lr()[0]
                    loop.set_description(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_num}/{dl}]")#, lr [{learnrate:.6f}]")
                    loop.set_postfix(loss=loss.item(), acc=avgacc/batch_num)
                #if batch_num % 100 == 0:
                    #wandb.log({"loss": loss, "accuracy": accuracy})

            eval = self.eval(valdata, valans)
            #scheduler.step()

            if eval > best_eval:
                best_eval = eval
                patience_count = 0 
            else:
                patience_count += 1

            if patience_count >= patience:
                print("Early stopping triggered. No improvement in validation accuracy.")
                break
    
    def pred(self, x):
        test_outputs = self(x)
        test_outputs = (test_outputs >= 0.5).squeeze().long().item()
        return test_outputs
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    once = False
    def printModel(self):
        if self.once == True:
            return
        print(self)
        self.once = True

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))

    def eval(self, X, y):
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        if len(X) < 1: return
        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=1)
        total_correct = 0
        total_samples = 0
        loop = tqdm(dataloader)
        for batch_inputs, batch_labels in loop:
            batch_labels = batch_labels.float().to(self.device)
            outputs = self(batch_inputs.to(self.device))
            predicted_labels = (outputs >= 0.5).long()
            correct = (predicted_labels == batch_labels.long()).sum().item()
            total_correct += correct
            total_samples += batch_labels.size(0)
        accuracy = total_correct / total_samples * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(0.1)
        
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)
        pex = self.pe[:x.size(0), :].to(self.device)
        x = x + pex
        return self.dropout.to(self.device)(x)


class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        #torch.nn.init.xavier_uniform_(self.feed_forward[0].weight)
        
    def forward(self, x):
        attended, _ = self.multihead_attention.to(self.device)(x, x, x)  # self-attention
        residual1 = self.norm1.to(self.device)(x + attended).to(self.device)
        
        fed_forward = self.feed_forward.to(self.device)(residual1)
        output = self.norm2.to(self.device)(residual1 + fed_forward)
        
        return output


import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Output a single value for binary classification
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, input_seq):
        self.to(self.device)
        input_seq = input_seq.to(self.device)
        embedded_seq = self.embedding.to(self.device)(input_seq)
        lstm_output, _ = self.lstm.to(self.device)(embedded_seq)
        last_hidden_state = lstm_output[:, -1, :].to(self.device)
        logits = self.fc.to(self.device)(last_hidden_state).squeeze(1)  # Remove the extra dimension
        predictions = torch.sigmoid(logits)  # Apply sigmoid activation function
        return predictions

    from torch.utils.data import DataLoader, TensorDataset

    def trainf(self, train_data, train_labels, epochs, batch_size, validation_ratio=0.1):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        X = train_data
        y = train_labels

        valdata = X[int(len(X)*(1-validation_ratio)):]
        valans = y[int(len(X)*(1-validation_ratio)):]

        X = X[:int(len(X)*(1-validation_ratio))]
        y = y[:int(len(X)*(1-validation_ratio))]

        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=batch_size)

        # Create DataLoaders for training and validation
        train_dataloader = DataLoader(dataloader, shuffle=True, batch_size=batch_size)
        #val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        dl = len(train_dataloader)
        total_samples = 0
        total_batches = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            loop = tqdm(dataloader, total=dl)
            batch_num = 0
            
            for batch_inputs, batch_labels in loop:
                batch_accuracy = 0
            # Training loop
            #for batch_inputs, batch_labels in train_dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                predictions = self(batch_inputs)
                loss = criterion(predictions.squeeze(), batch_labels.squeeze().float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                binary_predictions = (predictions > 0.5).long().unsqueeze(1)
                batch_accuracy = (binary_predictions == batch_labels).sum()
                batch_accuracy = batch_accuracy / batch_labels.size(0)
                epoch_accuracy += batch_accuracy
                total_samples += len(batch_inputs)
                batch_num += 1
                total_batches += 1
                if total_samples % 500 == 0:
                    #learnrate = scheduler.get_last_lr()[0]
                    loop.set_description(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_num}/{len(loop.iterable.dataset)/batch_size}]")#, lr [{learnrate:.6f}]")
                    loop.set_postfix(loss=loss.item(), acc=batch_accuracy.item())

            epoch_loss /= total_batches
            epoch_accuracy /= total_batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

    def trainf2(self, train_data, train_labels, epochs, batch_size):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        validation_ratio = 0.1

        X = train_data
        y = train_labels

        valdata = X[int(len(X)*(1-validation_ratio)):]
        valans = y[int(len(X)*(1-validation_ratio)):]

        X = X[:int(len(X)*(1-validation_ratio))]
        y = y[:int(len(X)*(1-validation_ratio))]

        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=batch_size)
        dl = len(dataloader)


        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = len(train_data) // batch_size
            num_batches = int(num_batches)


            loop = tqdm(dataloader, total=dl)
            batch_num = 0
            #for batch_inputs, batch_labels in loop:

            for i in range(num_batches):
                start_idx = int(i * batch_size)
                end_idx = int(start_idx + batch_size)

                batch_inputs = train_data[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx].to(self.device)

                optimizer.zero_grad()

                predictions = self(batch_inputs)
                loss = criterion(predictions.squeeze(), batch_labels.squeeze().float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                binary_predictions = (predictions > 0.5).long()
                batch_accuracy = torch.sum(binary_predictions == batch_labels).item() / batch_labels.size(0)
                epoch_accuracy += batch_accuracy

                epoch_loss /= num_batches
                epoch_accuracy /= num_batches

            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")
    
    def trainModel(self, X, y, num_epochs, batch_size, wandb=None, lr=0.001, weight_decay=0.001):
        z = (y == 0).sum()  # Number of negative instances
        e = (y == 1).sum()  # Number of positive instances
        f0 = z / (z + e)  # Frequency of negative instances
        f1 = e / (z + e)  # Frequency of positive instances
        pos_weight = f0 / f1
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        
        #criterion = FocalLoss(alpha=pos_weight, gamma=0.1) #nn.
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float)).to(self.device)
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float)).to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        patience = 6

        validation_ratio = 0.1

        valdata = X[int(len(X)*(1-validation_ratio)):]
        valans = y[int(len(X)*(1-validation_ratio)):]

        X = X[:int(len(X)*(1-validation_ratio))]
        y = y[:int(len(X)*(1-validation_ratio))]
        if len(X) < 1: return

        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=batch_size)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
        dl = len(dataloader)
        patience_count = 0 

        best_eval = 0
        for epoch in range(num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            loop = tqdm(dataloader, total=dl)
            batch_num = 0
            for batch_inputs, batch_labels in loop:
                optimizer.zero_grad()
                batch_labels = batch_labels.float().to(self.device)
                outputs = self(batch_inputs.to(self.device))
                #outputs = outputs.unsqueeze(0).transpose(0, 1)
                #outputs = outputs.round()
                #outputs = (outputs >= 0.5).float()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                predicted_labels = (outputs >= 0.5).long()
                correct = (predicted_labels == batch_labels.long()).sum().item()
                total_correct += correct
                total_samples += batch_labels.size(0)
                accuracy = total_correct / total_samples * 100
                batch_num += 1

                if total_samples % 50 == 0:
                    #learnrate = scheduler.get_last_lr()[0]
                    loop.set_description(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_num}/{dl}]")#, lr [{learnrate:.6f}]")
                    loop.set_postfix(loss=loss.item(), acc=accuracy)
                #if batch_num % 100 == 0:
                    #wandb.log({"loss": loss, "accuracy": accuracy})

            eval = self.eval(valdata, valans)
            #scheduler.step()

            if eval > best_eval:
                best_eval = eval
                patience_count = 0 
            else:
                patience_count += 1

            if patience_count >= patience:
                print("Early stopping triggered. No improvement in validation accuracy.")
                break
    
    def pred(self, x):
        test_outputs = self(x)
        test_outputs = (test_outputs >= 0.5).squeeze().long().item()
        return test_outputs
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    once = False
    def printModel(self):
        if self.once == True:
            return
        print(self)
        self.once = True

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))

    def eval(self, X, y):
        torch.no_grad()
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        dataloader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=1)
        total_correct = 0
        total_samples = 0
        loop = tqdm(dataloader)
        for batch_inputs, batch_labels in loop:
            batch_labels = batch_labels.float().to(self.device)
            outputs = self(batch_inputs.to(self.device))
            predicted_labels = (outputs >= 0.5).long()
            correct = (predicted_labels == batch_labels.long()).sum().item()
            total_correct += correct
            total_samples += batch_labels.size(0)
        accuracy = total_correct / total_samples * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy