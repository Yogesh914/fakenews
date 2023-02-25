from flask import Flask, render_template, request, jsonify
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
import re
import os
from nltk.stem.porter import PorterStemmer
from html_scraper import html_scraper

nltk.download('stopwords')

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder='.')
ps = PorterStemmer()
# Load model and vectorizer
classifier = 'rfmodel.pkl'
clfDTC = 'xgbmodel.pkl'
clfSVC = 'model2.pkl'
tfidfvect_filename = 'tfidfvect2.pkl'
classifier_path = os.path.join(basedir, classifier)
clfDTC_path = os.path.join(basedir, clfDTC)
clfSVC_path = os.path.join(basedir, clfSVC)
tfidfvect_path = os.path.join(basedir, tfidfvect_filename)


PCA = pickle.load(open(classifier_path, 'rb'))
DecisionTree = pickle.load(open(clfDTC_path, 'rb'))
SVC = pickle.load(open(clfSVC_path, 'rb'))
tfidfvect = pickle.load(open(tfidfvect_path, 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
def predict(text, ):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()

    score1 = PCA.decision_function(review_vect)
    proba1 = 1 / (1 + np.exp(-score1))
    prediction1 = PCA.predict(review_vect)

    score2 = DecisionTree.decision_function(review_vect)
    proba2 = 1 / (1 + np.exp(-score2))
    prediction2 = DecisionTree.predict(review_vect) 

    score3 = SVC.decision_function(review_vect)
    proba3 = 1 / (1 + np.exp(-score3))
    prediction3 = SVC.predict(review_vect)

    final_predict = (prediction1 + prediction2 + prediction3) / 3

    if final_predict < 0.5:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    final_prob = (proba1 + proba2 + proba3) / 3

    return prediction, final_prob[0]

@app.route('/', methods=['POST'])
def webapp():
    url = request.form['text']
    # If the URL doens't work, the text box is replaced with an error message
    try:
        text = html_scraper(url)
    except:
        return render_template('index.html', text = 'ERROR: URL doesn\'t work', result = 'FAKE', probability = "")

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