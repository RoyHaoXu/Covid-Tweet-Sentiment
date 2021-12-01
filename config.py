from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Flatten, GlobalMaxPooling1D
from keras_self_attention import SeqSelfAttention

# data cleaning
TEXT_COL = 'OriginalTweet'
SENTIMENT_COL = 'Sentiment'

# logistic regression
logistic_n_grams = [(1, 1), (1, 2)]
logistic_Cs = [1, 2]
logistic_penalty = ['l1', 'l2']

# svm
svm_n_grams = [(1, 1), (1, 2)]
svm_Cs = [1, 2]
svm_penalty = ['l1', 'l2']

# NN parameters
epochs = 10
batch_size = 256
max_length = 256
max_words = 50000
embedding_dim = 20

# LSTM models
model_lstm1 = Sequential()
model_lstm1.add(Embedding(max_words, embedding_dim))  # embedding layer
model_lstm1.add(SpatialDropout1D(0.2))  # dropout layer for regularization
model_lstm1.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
model_lstm1.add(Dense(5, activation='softmax'))  # softmax layer for classification
model_lstm1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_lstm2 = Sequential()
model_lstm2.add(Embedding(max_words, embedding_dim))  # embedding layer
model_lstm2.add(SpatialDropout1D(0.1))  # dropout layer for regularization
model_lstm2.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
model_lstm2.add(Dense(5, activation='softmax'))  # softmax layer for classification
model_lstm2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_lstm3 = Sequential()
model_lstm3.add(Embedding(max_words, embedding_dim))  # embedding layer
model_lstm3.add(SpatialDropout1D(0.2))  # dropout layer for regularization
model_lstm3.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
model_lstm3.add(Dense(5, activation='softmax'))  # softmax layer for classification
model_lstm3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_models = [model_lstm1, model_lstm2, model_lstm3]

# BLSTM models
model_blstm1 = Sequential()
model_blstm1.add(Embedding(max_words, embedding_dim))
model_blstm1.add(SpatialDropout1D(0.2))
model_blstm1.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model_blstm1.add(Dense(5, activation='softmax'))
model_blstm1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_blstm2 = Sequential()
model_blstm2.add(Embedding(max_words, embedding_dim))
model_blstm2.add(SpatialDropout1D(0.1))
model_blstm2.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model_blstm2.add(Dense(5, activation='softmax'))
model_blstm2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_blstm3 = Sequential()
model_blstm3.add(Embedding(max_words, embedding_dim))
model_blstm3.add(SpatialDropout1D(0.2))
model_blstm3.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model_blstm3.add(SeqSelfAttention(attention_activation='sigmoid'))
model_blstm3.add(GlobalMaxPooling1D())
model_blstm3.add(Dense(5, activation='softmax'))
model_blstm3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

blstm_models = [model_blstm1, model_blstm2, model_blstm3]

# Bert
bert_model = {'model_name': 'model1',
              'epochs': 10,
              'max_length': 256,
              'batch_size': 8}
