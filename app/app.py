import streamlit as st
import pickle
from keras.models import load_model


def predict_label(raw_texts, model_path='model/model.h5', tokenizer_path='model/tokenizer.pickle'):
    if not raw_texts:
        return {}
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    input = tokenizer.texts_to_sequences([raw_texts])
    predict = list(model.predict(input)[0])
    labels = ['Extremely Negative', 'Extremely Positive', 'Negative', 'Neutral', 'Positive']

    prediction = labels[predict.index(max(predict))]
    return {'predicted label': prediction,
            'probabilities': {l: p for
                              l, p in
                              zip(labels,
                                  predict)}}


st.header("Covid Tweet Sentiment Detection")
raw_texts = st.text_input("Please enter tweet:")
pred = predict_label(raw_texts)
st.write(pred)
