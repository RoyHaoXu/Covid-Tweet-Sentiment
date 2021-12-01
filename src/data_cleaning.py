import pandas as pd
import re


def clean_text(text):
    # lower
    text = text.lower()
    # regex
    remove_special_characters = re.compile('[^0-9a-z +]')
    remove_numbers = re.compile('[0-9]+')
    # clean
    text = re.sub(remove_special_characters, ' ', text)
    text = re.sub(remove_numbers, ' ', text)
    return text.strip()


def clean_data(data_path, output_path, text_col, sentiment_col):
    # read data
    data = pd.read_csv(data_path, encoding='latin_1')
    # keep relevant cols
    data = data[[text_col, sentiment_col]]
    # clean text
    data[text_col] = data[text_col].apply(lambda e: clean_text(e))

    data.to_csv(output_path, index=False)


