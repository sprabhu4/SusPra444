#Text Processing related libraries
import re
import string
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from HanTa import HanoverTagger as ht

import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

app = Flask(__name__)
model = load("final_model.joblib")

def text_cleaning(text):
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    text = re.sub(r"[^A-Za-zäöüÄÖÜß \t]", "", text)
    text= text.lower()
    temp = []
    stop_words = set(stopwords.words('german'))
    stop_words.add("fuer")
    word_tokens= nltk.tokenize.WordPunctTokenizer().tokenize(text)
    for w in word_tokens:
        if w not in stop_words: #Removing stop words
            temp.append([tagger.analyze(w)[0]]) #The method analyze gives the lemma of a word.
    text = [" ".join(i) for i in temp]
    text = " ".join(text)
    return text
    
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    mapping = ['ch','cnc','ct', 'ft', 'mr', 'pkg']
    text1 = request.form['Input Text']
    text2 = [text_cleaning(text1)]
    prediction = model.predict(text2)

    output = mapping[int(prediction)]

    return render_template('index.html', prediction_text='Label should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)