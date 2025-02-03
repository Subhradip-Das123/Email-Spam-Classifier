import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# cv = CountVectorizer()
# tfidf = TfidfVectorizer
#
# ps = PorterStemmer()
#
#
# def transform_text(text):
#     text = text.lower()
#     text = text.split()
#
#     y=[]
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]
#     y=[]
#
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#
#     text = y[:]
#     y=[]
#
#     for i in text:
#         y.append(ps.stem(i))
#
#     return " ".join(y)
#
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))
#
# st.title("Email/SMS Spam Classifier")
# input_sms = st.text_input("Enter the message")
#
# #1. preprocess
# transform_sms = transform_text(input_sms)
# #2. vectorize
# vector_input = tfidf.transform([transform_sms])
# #3. predict
# result = model.predict(vector_input)[0]
# #4.
# if result == 1:
#     st.header("Spam")
# else:
#     st.header("Not Spam")

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # Create a copy of y
    y = []  # Clear y after using it

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y = []  # Clear y again

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
loaded_tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = loaded_tfidf.transform([transform_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Output result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

