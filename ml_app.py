import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pickle import dump, load

st.markdown("<h3 style='text-align: center'>🙂😐😔Product Sentiment Analysis App🙂😐😔</h3>", unsafe_allow_html=True)

def preprocess(tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",tweet)

    letters_only = letters_only.lower()

    words = letters_only.split()

    words = [w for w in words if not w in stopwords.words("english")]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    clean_sentence = " ".join(words)

    return clean_sentence



def predict(tweet):

    vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))

    classifier = load(open('pickle/logit_model.pkl', 'rb'))

    clean_tweet = preprocess(tweet)

    clean_tweet_encoded = vectorizer.transform([clean_tweet])

    tweet_input = clean_tweet_encoded.toarray()

    prediction = classifier.predict(tweet_input)

    return prediction



def main():

#    st.image("img/twitter_img.jpg", use_column_width = True)

    tweet = st.text_input('Enter Your Review about the Product')

    prediction = predict(tweet)

    st.button('predict')

    if(tweet):

        st.markdown("<h6 style='color: red'>Review of the product is:</h1>", unsafe_allow_html=True)
        if(prediction == 'negative'):
            st.write("Negative Review 😔")
        elif(prediction == 'neutral'):
            st.write("Neutral  Review 😐")
        else:
            st.write("Positive Review 🙂")



if(__name__ == '__main__'):
    main()