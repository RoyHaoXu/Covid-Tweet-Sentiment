from sklearn.svm import LinearSVC
import pandas as pd

from src.helpers import tfidf_vectorizer, model_evaluation


def model_tuning_svm(training_path, testing_path, text_col, sentiment_col, logistic_n_grams, logistic_Cs, logistic_penalty, output_path):
    # get data
    training = pd.read_csv(training_path)
    testing = pd.read_csv(testing_path)
    X_train = training[text_col]
    X_test = testing[text_col]
    y_train = training[sentiment_col]
    y_test = testing[sentiment_col]

    # hyper-parameter tuning
    model_tuning_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'])

    for n_gram in logistic_n_grams:
        # get tfidf vector matrix
        X_train_tfidf, X_test_tfidf = tfidf_vectorizer(X_train, X_test, n_gram)
        # train and test models
        for c in logistic_Cs:
            for p in logistic_penalty:
                model_name = 'model_p={0}_c={1}_ngram={2}'.format(p, c, n_gram)
                print(model_name)
                model = LinearSVC(C=c, penalty=p, random_state=1, dual=False)
                svm = model.fit(X_train_tfidf, y_train)
                accuracy, precision, recall, f1 = model_evaluation(svm, X_test_tfidf, y_test)
                model_tuning_df.loc[model_name] = [accuracy, precision, recall, f1]

    model_tuning_df.to_csv(output_path)
