import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, Embedding, LSTM, Activation, Dropout, Concatenate, Input, Attention, Layer, Conv1D, GlobalMaxPooling1D, concatenate, BatchNormalization
from matplotlib import pyplot as plt
import Files as files
import json
from keras.regularizers import L1L2, l2
"""
def GenerateAndCompileModel(num, maxLenPost):
    model = modelText(num, maxLenPost)
    model.compile(optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['binary_accuracy'])#['accuracy'])#['categorical_accuracy'])
    return model
"""

def modelText(num, numChar, cvNum, maxLen, maxLenChar, maxVocLen):
    ScaleTest = 1
    embedding_dim = 10

    #input1 = Input(shape=(None, ))
    #input2 = Input(shape=(maxLenChar,))
    input3 = Input(shape=(None,))

    #embedding1 = Embedding(input_dim=num, output_dim=embedding_dim)(input1)
    embedding2 = Embedding(input_dim=cvNum, output_dim=embedding_dim)(input3)

    #merged = Concatenate(axis=1)([embedding1, embedding2])#input2, input3])

    outLen = 1
    LSTMin = 8
    numDense = 1
    inLen = 4#cvNum + num

    #inLen = round((maxLen+maxLenChar+maxVocLen)*2/3)
    #round((maxLen+cvNum)/24)
    #inLenOriginal = maxLen+maxLenChar+maxVocLen
    inLenOriginal = cvNum + num

    #x = Embedding(inLenOriginal, embedding_dim)(merged)
    x = embedding2#merged
    x = (Bidirectional(LSTM(LSTMin, return_sequences=False)))(x)#, kernel_regularizer=L1L2(0.0, 0.01))))(x)#, kernel_regularizer=L1L2(l1=0.0001, l2=0.0)))(x)

    for i in range(numDense):
        x = Dropout(0.4)(x)
        x = Dense(inLen, activation=tf.nn.relu)(x)

    #BatchNormalization(),
    #Activation(tf.nn.relu),
    #x = (Bidirectional(LSTM(LSTMin, return_sequences=True)))(x)#, kernel_regularizer=L1L2(0.0, 0.01))))(x)#, kernel_regularizer=L1L2
    #x = Dropout(0.2)(x)

    
    inLen = round(inLen/2)
    numDense = 2*numDense

    for i in range(numDense):
        x = Dropout(0.4)(x)
        x = Dense(inLen, activation=tf.nn.relu)(x)

    #BatchNormalization(),
    #Activation(tf.nn.relu),

    inLen = round(inLen/2)
    numDense = 2*numDense

    
    for i in range(numDense):
        x = Dropout(0.4)(x)
        x = Dense(inLen, activation=tf.nn.relu)(x)
    
    inLen = round(inLen/2)

    #x = Dropout(0.2)(x)
    #x = Dense(inLen, activation=tf.nn.relu)(x)
    #x = AttentionLayer()(x)
    #x = Dropout(0.2)(x)
    #x = Dense(inLen*ScaleTest, activation=tf.nn.relu)(x)
    x = Dense(outLen, activation='sigmoid')(x)

    model = keras.models.Model(inputs=input3, outputs=x)
    model.summary()
    return model

def improved_model(num, numChar, cvNum, maxLen, maxLenChar, maxVocLen):
    ScaleTest = 1
    embedding_dim = 5
    dropout_rate = 0.4
    num_filters = 256
    filter_sizes = [3, 4, 5, 6]
    inLen = 256

    numDense = 2
    
    
    input1 = Input(shape=(None,), name='input1')
    input3 = Input(shape=(None,), name='input3')

    embedding1 = Embedding(input_dim=num, output_dim=embedding_dim)(input1)

    embedding2 = Embedding(input_dim=cvNum, output_dim=embedding_dim)(input3)
    merged = Concatenate(axis=1)([embedding1, embedding2])#input2, input3])
    x = merged

    # Apply convolutional layers with varying filter sizes
    conv_blocks = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters=num_filters,
                    kernel_size=filter_size,
                    padding='valid')(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(dropout_rate)(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    x = concatenate(conv_blocks)

    #inLen = round(inLen/2)
    #numDense = 2*numDense

    for i in range(numDense):
        x = Dropout(dropout_rate)(x)
        x = Dense(inLen, 
                  activation='relu', 
                  kernel_regularizer=l2(0.01))(x)
    
    inLen = round(inLen/2)
    numDense = 2*numDense
    
    for i in range(numDense):
        x = Dropout(dropout_rate)(x)
        x = Dense(inLen, 
                  activation='relu', 
                  kernel_regularizer=l2(0.01))(x)
    
    inLen = round(inLen/2)
    numDense = 2*numDense
    
    for i in range(numDense):
        x = Dropout(dropout_rate)(x)
        x = Dense(inLen, 
                  activation='relu', 
                  kernel_regularizer=l2(0.01))(x)

    # Output layer
    x = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=[input1, input3], outputs=x)
    model.summary()
    #exit()
    return model


def model2(num, numChar, cvNum, maxLen, maxLenChar, maxVocLen):
    #input1 = Input(shape=(None, ))
    #input2 = Input(shape=(maxLenChar,))
    input3 = Input(shape=(None,))
    #merged = Concatenate(axis=1)([input1, input3])#input2, input3])

    # Define the embedding layer
    embedding_dim = 1
    embedding = Embedding(cvNum, embedding_dim)(input3)

    # Define the LSTM layer
    hidden_dim = 16
    lstm = LSTM(hidden_dim, return_sequences=True)(embedding)

    #test = tf.keras.layers.MaxPooling1D()(lstm)
    #test = tf.keras.layers.UpSampling1D(size=0.5)(test)

    # Define the attention layer
    attention = AttentionLayer()(lstm)
    #lstm = LSTM(hidden_dim, return_sequences=True)(attention)
    # Define the output layer
    dropout_rate = 0.5
    output_dim = 2
    dropout = Dropout(dropout_rate)(attention)
    #dropout = LSTM(hidden_dim, return_sequences=False)(dropout)
    outputs = Dense(output_dim, activation='softmax')(dropout)

    # Define the model
    model = keras.models.Model(inputs=input3, outputs=outputs)
    model.summary()
    return model

def model3(num, numChar, cvNum, maxLen, maxLenChar, maxVocLen):
    input3 = Input(shape=(None,))
    embedding = Embedding(cvNum, 100)(input3)
    embedding = keras.layers.GlobalAveragePooling1D()(embedding)
    #embedding = keras.layers.Flatten()(embedding)
    x = Dense(32, activation=tf.nn.relu)(embedding)
    outputs = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=input3, outputs=outputs)
    model.summary()
    return model
    
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True, name='weightsName')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute the attention scores
        e = tf.keras.backend.dot(inputs, self.w)
        attention_scores = tf.keras.backend.softmax(e, axis=1)

        # Weight the input vectors with the attention scores
        weighted_input = inputs * attention_scores

        # Compute the context vector
        context_vector = tf.keras.backend.sum(weighted_input, axis=1)

        return context_vector

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def modelTextAlt(num, numChar, cvNum, maxLen, maxLenChar, maxVocLen):
    ScaleTest = 1
    embedding_dim = 50
    input1 = Input(shape=(maxLen, ))
    #input2 = Input(shape=(maxLenChar,))
    input3 = Input(shape=(maxVocLen,))
    merged = Concatenate(axis=1)([input1, input3])
    x = Embedding(cvNum+num, embedding_dim, input_length=maxLen+maxVocLen)(merged)
    x = Bidirectional(LSTM(16*ScaleTest, return_sequences=True))(x)#, kernel_regularizer=L1L2(l1=0.0001, l2=0.0)))(x)#, recurrent_dropout=0.2))
    #BatchNormalization(),
    #Activation(tf.nn.relu),
    x = Dropout(0.2)(x)
    x = Dense(8*ScaleTest)(x)
    x = Activation(tf.nn.relu)(x)
    x = Bidirectional(LSTM(8*ScaleTest, return_sequences=True))(x)#, kernel_regularizer=L1L2(l1=0.0001, l2=0.0)))(x)
    #BatchNormalization(),
    #Activation(tf.nn.relu),
    x = Dense(8*ScaleTest/2)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.15)(x)
    x = Dense(2*ScaleTest/2, activation=tf.nn.relu)(x)
    x = Dense(1, activation=tf.nn.sigmoid)(x)
    model = keras.models.Model(inputs=[input1, input3], outputs=x)
    model.summary()
    return model