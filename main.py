import nltk

nltk.download('punkt')
nltk.download('stopwords')

import pandas
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from pickle_files import vectorizer, model_mnb


def remove_pattern(text):
    return re.sub(r"\([^()]*\)", '', text, 0, re.MULTILINE)


def take_only_alphabets(text):
    return re.sub(r"[^a-zA-Z]", ' ', text, 0, re.MULTILINE)


def convert_to_tokenize_words_without_stopwords(x):
    temp = []
    for word in word_tokenize(x):
        word = word.lower()
        if word not in stopwords.words('english'):
            temp.append(word)
    return temp


def stem_words(words):
    temp = ' '
    for word in words:
        temp += SnowballStemmer("english").stem(word) + ' '

    return temp


def sentiment_analyzer(user_text):

    if user_text == '':
        return ''

    txt = remove_pattern(user_text)
    txt = take_only_alphabets(txt)
    txt = convert_to_tokenize_words_without_stopwords(txt)
    txt = stem_words(txt)

    txt_vect = vectorizer.transform(pandas.Series(txt)).toarray()
    prediction = model_mnb.predict(txt_vect)

    if prediction == 1:
        return 'Positive'
    else:
        return 'Negative'
