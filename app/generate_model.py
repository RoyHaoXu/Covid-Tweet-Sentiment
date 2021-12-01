import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Flatten, GlobalMaxPooling1D
import pickle


def model_tuning_lstm(model, training_path, testing_path, text_col, sentiment_col,
                      epochs, batch_size, max_length, max_words, model_path, tokenizer_path):
    # data
    training = pd.read_csv(training_path)
    testing = pd.read_csv(testing_path)
    all_data = pd.concat([training, testing])

    # tokenizer
    tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(all_data[text_col].values)

    # data prepare
    X_train = tokenizer.texts_to_sequences(training[text_col].values)
    X_train = pad_sequences(X_train, maxlen=max_length)

    y_train = pd.get_dummies(training[sentiment_col]).values

    # training
    with tf.device('/cpu:0'):  # used cpu as tf gpu performance is really bad for my M1 chip
        _ = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

    # save
    model.save(model_path)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    model_lstm2 = Sequential()
    model_lstm2.add(Embedding(50000, 20))  # embedding layer
    model_lstm2.add(SpatialDropout1D(0.2))  # dropout layer for regularization
    model_lstm2.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
    model_lstm2.add(Dense(5, activation='softmax'))  # softmax layer for classification
    model_lstm2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_tuning_lstm(model_lstm2, '../data/Corona_NLP_train_cleaned.csv', '../data/Corona_NLP_test_cleaned.csv', 'OriginalTweet', 'Sentiment',
                      10, 256, 256, 50000, './model/model.h5', './model/tokenizer.pickle')