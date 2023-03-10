from flask import Flask, render_template, request, jsonify, redirect
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
import re
import os
from nltk.stem.porter import PorterStemmer
from html_scraper import html_scraper
import csv

nltk.download('stopwords')

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder='.')
ps = PorterStemmer()
# Load model and vectorizer
rfmodel = 'rfmodel.pkl'
Bmodel = 'modelbnb.pkl'
DTCmodel = 'modelDTC.pkl'
PCAmodel = 'model2.pkl'
tfidfvect_filename = 'tfidfvect2.pkl'
Bpath = os.path.join(basedir, Bmodel)
DTCpath = os.path.join(basedir, DTCmodel)
PCApath = os.path.join(basedir, PCAmodel)
tfidfvect_path = os.path.join(basedir, tfidfvect_filename)
rfpath = os.path.join(basedir, rfmodel)

B = pickle.load(open(Bpath, 'rb'))
DCT = pickle.load(open(DTCpath, 'rb'))
PCA = pickle.load(open(PCApath, 'rb'))
tfidfvect = pickle.load(open(tfidfvect_path, 'rb'))
rf = pickle.load(open(rfpath, 'rb'))
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

    score1 = B.decision_function(review_vect)
    proba1 = 1 / (1 + np.exp(-score1))
    prediction1 = B.predict(review_vect)

    prediction2 = DCT.predict(review_vect) 

    score3 = PCA.decision_function(review_vect)
    proba3 = 1 / (1 + np.exp(-score3))
    prediction3 = PCA.predict(review_vect)

    score4 = rf.predict_proba(review_vect)
    proba4 = 1 / (1 + np.exp(-score4))
    prediction4 = rf.predict(review_vect)

    final_predict = (prediction1 + prediction2 + prediction3 + prediction4) / 4

    if final_predict < 0.5:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    final_prob = (proba1 + proba3) / 2

    return prediction, final_prob

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

@app.route('/feedback', methods=['POST'])
def handle_feedback():
  feedback = request.json['feedback']
  text = request.json['text']
  
  with open('feedback.csv', mode='a', newline='') as feedback_file:
    feedback_writer = csv.writer(feedback_file)
    feedback_writer.writerow([text, int(feedback == 'real')])
  
  return redirect('/')

if __name__ == "__main__":
    app.run()