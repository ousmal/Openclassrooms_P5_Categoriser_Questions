import flask
import re
import os
import numpy as np
import joblib
import pandas as pd
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from ast import literal_eval

# from Multi_labelling import y_test_inversed, y_test_predicted_labels_tfidf_rfc

df = pd.read_csv("C:/Users/ousma/PycharmProjects/Project5/data_with_tags_frequent.csv", converters={"Title": literal_eval,
                                                  "Body": literal_eval,
                                                  "Tags": literal_eval})

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

X = df['Title'] + df['Body']
Y = df["Tags"]


'''
function to clean a list containing the words of the sentence.
- Replace capitals letters
- remove punctuation
- remove digits
arg : 
    t : a list of strings 
'''



def ValuePredictor(sentence):



    keyword_model = joblib.load("C:/Users/ousma/PycharmProjects/Project5/model_pipeline.pkl")
    transformer = joblib.load("C:/Users/ousma/PycharmProjects/Project5/mlb_transformer.pkl")

    y_test_predicted = keyword_model.predict(text)

    # Inverse transform
    y_test_pred_inversed = transformer \
        .inverse_transform(y_test_predicted)

    result = y_test_pred_inversed
    l = len(result)
    #  for i in range(l):
    #     print(result[i])
    return result


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence1 = request.form['title']
        sentence2 = request.form['body']

        to_predict_list1 = sentence1 + ' ' + sentence2

        #         to_predict_list = preprocess(to_predict_list1)
        prediction = ValuePredictor(to_predict_list1)
        #         prediction = str(result)
        return render_template('predict.html', prediction=prediction)


#     sentence1 = 'when I run command php artisan optimize  this error appeares'
#     sentence2 = 'Then I have to go to bootstrap cache directory and need to delete files then error disappear. What is the issue how I   could solve it ?'
#     sentence = sentence1 + ' ' + sentence2
#     res = ValuePredictor(sentence)
#     return f" Votre question : <br><br> {sentence1} <br><br> {sentence2} <br><br> Le(s) Tag(s) détecté(s) pour votre question : {str(res)}"

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)