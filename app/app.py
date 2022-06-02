import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
	
	if request.method == 'POST':
		Enter_Comment = request.form['Enter_Comment']

		data = [[str(Enter_Comment)]]
		
		vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))

		classifier = load(open('pickle/logit_model.pkl', 'rb'))
		
		prediction = classifier.predict(data)[0]
	return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)