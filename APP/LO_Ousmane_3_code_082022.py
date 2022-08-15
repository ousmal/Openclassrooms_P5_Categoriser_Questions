import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from nlpk_module import print_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from time import time
from pickle import dump
import joblib

######################################
###########DATA PREPARATION###########
######################################

print("Loading dataset")
t0 = time()
path = "C:/Users/ousma/PycharmProjects/Project5/data_with_tags_frequent.csv"
data = pd.read_csv(path, encoding="utf-8")

print(data.head(3))
print("Dataset Loaded")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")
print("Preprocessing...")
t0 = time()
data.dropna(inplace=True)

tags = data["Tags"].apply(lambda x: x[0:].split(','))

docs = data["Title"].values \
       + " " \
       + data["Body"].values

mlb = MultiLabelBinarizer()
mlb.fit(tags)
tags_mlb = mlb.transform(tags)

print("Number of Tags:", len(mlb.classes_))

dump(mlb, open("C:/Users/ousma/PycharmProjects/Project5/mlb_transformer.pkl",'wb'))

########################################
############SPLITTING DATA##############
########################################

X = docs.copy()
y = tags_mlb.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.25)

print("Preprocessing finished!")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")

########################################
#########Feature extraction with USE####
########################################

########################################
#########PIPELINE & FITTING#############
########################################

#print('Training...')
t0=time()

final_model = Pipeline(steps=[('tfidf',
                 TfidfVectorizer(max_df=0.01, max_features=12000,
                                 min_df=0.001)),
                ('clf',
                 OneVsRestClassifier(estimator=LogisticRegression(C=100,
                                                                  random_state=1,
                                                                  solver='sag'),
                                     n_jobs=-1))])
final_model.fit(X_train, y_train)

print("Training finished!")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")

########################################
##############PREDICTION ###############
########################################

print("Prediction...")
t0 = time()

y_pred = final_model.predict(X_test)
print("Prediction finished!")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")

print("Evaluation of the model:\n ")
print("Prediction on a new post")

my_title = "I update datum plot matlab suppose want update plot new datum method choose set xdatasource property name update variable call refreshdata erase original plot call plot command use set xdata"
my_body = "message broker vs mom message orient middleware little confused difference message broker e g rabbitmq message orientate middleware not find much info apart wikipedia search mom find info amqp state protocol mom mean mom also read rabbitmq implement ampq protocol make rabbitmq messsage broker message broker mom thing hope unravel confusion thank"

text = my_title + my_body
text = [text]

def Performance_score(feature, name_model, model, X_test, y_true):
    y_pred = model.predict(X_test)

    temp_df = {'Score': ['hamming_loss', "Jaccard", "F1-score",
                                      ]}
    scores = []
    scores.append(metrics.hamming_loss(y_true, y_pred))
    scores.append(metrics.jaccard_score(y_true,
                                        y_pred,
                                        average='micro'))
    scores.append(metrics.f1_score(y_pred,
                                   y_true,
                                   average='micro'))

    temp_df[name_model + ' + ' + feature] = scores
    return temp_df
t0 = time()
Performance_score('tfidf','LogisticRegression',final_model,X_test,y_test)
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")

######################################
############MODEL SAVING##############
######################################

print("Saving model...")
t0 = time()

joblib.dump(final_model, "C:/Users/ousma/PycharmProjects/Project5/model_pipeline.pkl")
print("Model saved!")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")

######################################
###########MODEL TRIAL################
######################################

print("Model loading...")

final_model = joblib.load("C:/Users/ousma/PycharmProjects/Project5/model_pipeline.pkl")
transformer = joblib.load("C:/Users/ousma/PycharmProjects/Project5/mlb_transformer.pkl")

print("Model loaded!")
t0 = time()
my_prediction = final_model.predict(text)
my_prediction = mlb.inverse_transform(my_prediction)
print("done in %0.3fs." % (time() - t0))

my_prediction = [x for x in my_prediction if x != ()]
print("My tags", list(set(my_prediction)))