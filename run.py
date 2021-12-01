import argparse
import config

from src.data_cleaning import clean_data
from src.logistic_regression import model_tuning_logistic
from src.svm import model_tuning_svm
from src.lstm import model_tuning_lstm
from src.blstm import model_tuning_blstm
from src.bert import model_tuning_bert

if __name__ == '__main__':

    # Parse args and run corresponding pipeline
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('step', help='Which step to run',
                        choices=['clean_data', 'logistic', 'svm', 'lstm', 'blstm', 'bert'])
    parser.add_argument('--data_path', help='Path to raw data')
    parser.add_argument('--output_path', help='Path to store cleaned data')
    parser.add_argument('--training_data', help='Path to train data')
    parser.add_argument('--testing_data', help='Path to test data')

    args = parser.parse_args()

    # clean data
    if args.step == 'clean_data':
        clean_data(args.data_path, args.output_path, text_col=config.TEXT_COL, sentiment_col=config.SENTIMENT_COL)

    # fit models
    # logistic regression
    elif args.step == 'logistic':
        model_tuning_logistic(args.training_data, args.testing_data,
                              text_col=config.TEXT_COL, sentiment_col=config.SENTIMENT_COL,
                              logistic_n_grams=config.logistic_n_grams,
                              logistic_Cs=config.logistic_Cs,
                              logistic_penalty=config.logistic_penalty,
                              output_path=args.output_path)

    elif args.step == 'svm':
        model_tuning_svm(args.training_data, args.testing_data,
                         text_col=config.TEXT_COL, sentiment_col=config.SENTIMENT_COL,
                         logistic_n_grams=config.svm_n_grams,
                         logistic_Cs=config.svm_Cs,
                         logistic_penalty=config.svm_penalty,
                         output_path=args.output_path)

    elif args.step == 'lstm':
        model_tuning_lstm(config.lstm_models,
                          args.training_data,
                          args.testing_data,
                          config.TEXT_COL,
                          config.SENTIMENT_COL,
                          config.epochs,
                          config.batch_size,
                          config.max_length,
                          config.max_words,
                          args.output_path)

    elif args.step == 'blstm':
        model_tuning_blstm(config.blstm_models,
                           args.training_data,
                           args.testing_data,
                           config.TEXT_COL,
                           config.SENTIMENT_COL,
                           config.epochs,
                           config.batch_size,
                           config.max_length,
                           config.max_words,
                           args.output_path)

    elif args.step == 'bert':
        model_tuning_bert(config.bert_model['model_name'],
                          args.training_data,
                          args.testing_data,
                          config.TEXT_COL,
                          config.SENTIMENT_COL,
                          config.bert_model['batch_size'],
                          config.bert_model['epochs'],
                          config.bert_model['max_length'])