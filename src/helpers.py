import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def tfidf_vectorizer(X_train, X_test, ngram_range):
    # initialize  the tfidf vectorized
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=ngram_range, stop_words='english')

    # fit it to X_train
    tfidf_fit = vectorizer.fit_transform(X_train)

    # transform
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # return the transformed matrix
    return X_train_tfidf, X_test_tfidf


def model_evaluation(model, X_test, y_test):
    pred = model.predict(X_test)
    precision, recall, f1, accuracy = precision_score(y_test, pred, average='weighted'), \
                                      recall_score(y_test, pred, average='weighted'), \
                                      f1_score(y_test, pred, average='weighted'), \
                                      accuracy_score(y_test, pred)
    return accuracy, precision, recall, f1


def model_evaluation_nn(model, X_test, y_test):
    pred = np.argmax(model.predict(X_test), axis=1)
    precision, recall, f1, accuracy = precision_score(y_test, pred, average='weighted'), \
                                      recall_score(y_test, pred, average='weighted'), \
                                      f1_score(y_test, pred, average='weighted'), \
                                      accuracy_score(y_test, pred)
    return accuracy, precision, recall, f1


def categorize(vec):
    for i in range(len(vec)):
        if vec[i] == 1:
            return i
