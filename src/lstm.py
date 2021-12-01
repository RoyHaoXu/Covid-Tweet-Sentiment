import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from .helpers import model_evaluation_nn, categorize


def model_tuning_lstm(models, training_path, testing_path, text_col, sentiment_col,
                      epochs, batch_size, max_length, max_words, output_path):
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
    X_test = tokenizer.texts_to_sequences(testing[text_col].values)
    X_test = pad_sequences(X_test, maxlen=max_length)

    y_train = pd.get_dummies(training[sentiment_col]).values
    y_test = pd.get_dummies(testing[sentiment_col]).values
    y_test = np.apply_along_axis(categorize, 1, y_test)

    # training
    model_tuning_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    for i, model in enumerate(models):
        with tf.device('/cpu:0'):  # used cpu as tf gpu performance is really bad for my M1 chip
            _ = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

        # evaluation
        model_name = 'lstm{0}'.format(i)
        accuracy, precision, recall, f1 = model_evaluation_nn(model, X_test, y_test)
        model_tuning_df.loc[model_name] = [accuracy, precision, recall, f1]

    model_tuning_df.to_csv(output_path)



