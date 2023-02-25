from flask import Flask, render_template, request, jsonify
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
import re
import os
from nltk.stem.porter import PorterStemmer
from html_scraper import html_scraper

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder='.')
ps = PorterStemmer()
# Load model and vectorizer
model_filename = 'model2.pkl'
tfidfvect_filename = 'tfidfvect2.pkl'
model_path = os.path.join(basedir, model_filename)
tfidfvect_path = os.path.join(basedir, tfidfvect_filename)


model = pickle.load(open(model_path, 'rb'))
tfidfvect = pickle.load(open(tfidfvect_path, 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    score = model.decision_function(review_vect)
    proba = 1 / (1 + np.exp(-score))
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction, proba[0]

@app.route('/', methods=['POST'])
def webapp():
    url = request.form['text']
    # If the URL doens't work, the text box is replaced with an error message
    try:
        text = html_scraper(url)
    except:
        return render_template('index.html', text = 'ERROR: URL doesn\'t work', result = "", probability = "")

    prediction, probability = predict(text)

    probability = f"{probability.max()*100:.2f}%"

    return render_template('index.html', text=text, result=prediction, probability=probability)

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)
if __name__ == "__main__":
    app.run()