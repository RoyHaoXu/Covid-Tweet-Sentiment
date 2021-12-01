### Northwestern University
#### MSiA-490 Fall 2021
#### Covid tweet sentiment classification
#### Hao Xu


### Project Topic
In this project, I trained, fine-tuned, and evaluated various multi-class text classification models (logistic regression, 
SVM, LSTM, bidirectional LSTM, and bidirectional LSTM etc.) Each model was evaluated using metrics such as accuracy, precision, recall, and F-1 score. The best model was also productized as a web app to take review text as input and the output the 
corresponding predicted sentiment.

### Dataset
The dataset chosen for this project is covid tweets data from [Covid Tweets](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification). 
The raw dataset contains covid related tweets pulled from Twitter. The data is manual tagged with the sentiment and the names and usernames have been given codes to avoid any privacy concerns. There are 41157 records in the training dataset and 3798 records in the testing dataset. There are in total 5 classes ('Extremely Negative', 'Extremely Positive', 'Negative', 'Neutral', 'Positive') so it will be a multiclass classification problem. 

### Web App for covid tweet sentiemnt detection

To run the web app, please clone the Github repo and run the following command in terminal.

```shell script
pip install -r requirements.txt
cd app
python generate_model.py
streamlit run app.py
```

A browser will automatically pop up with the API UI. Press  CTRL+C to at any time to quit.


### Model Results

For the traditional models, neither SVM nor logistic regression yields decent results which are inline with expectation. NN based model has much better results in compare with traditional models. Bidirectional LSTM yields the best results. 

I did also set up BERT model but it took a much longer time to 
train, compared to LSTM and other traditional machine learning models. With limited computing resources I decided not to use the bert model for production, but the training pipeline is set up in the source code. I believe if hardware permits, training BERT model may lead 
to an even better performance.


### Reproduce models

#### Clean data
```shell script
python run.py clean_data --data_path="data/Corona_NLP_test.csv" --output_path="data/Corona_NLP_test_cleaned.csv" 
python run.py clean_data --data_path="data/Corona_NLP_train.csv" --output_path="data/Corona_NLP_train_cleaned.csv"
```

#### Logistic Regression

```shell script
python run.py logistic --training_data="data/Corona_NLP_train_cleaned.csv" --testing_data="data/Corona_NLP_test_cleaned.csv" --output_path="results/logistic.csv"
```

#### Support Vector Machine

```shell script
python run.py svm --training_data="data/Corona_NLP_train_cleaned.csv" --testing_data="data/Corona_NLP_test_cleaned.csv" --output_path="results/svm.csv"
```

#### LSTM

```shell script
python run.py lstm --training_data="data/Corona_NLP_train_cleaned.csv" --testing_data="data/Corona_NLP_test_cleaned.csv" --output_path="results/lstm.csv"
```


#### BLSTM

```shell script
python run.py blstm --training_data="data/Corona_NLP_train_cleaned.csv" --testing_data="data/Corona_NLP_test_cleaned.csv" --output_path="results/blstm.csv"
```

#### BERT
```shell script
python run.py bert --training_data="data/Corona_NLP_train_cleaned.csv" --testing_data="data/Corona_NLP_test_cleaned.csv" 
```
