import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from nlpk_module import normalize_corpus, tok, print_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
from pickle import dump
import joblib

######################################
###########DATA PREPARATION###########
######################################

print("Loading dataset")
t0 = time()
path= "/content/gdrive/My Drive/P5/data_with_tags_more_frequent.csv"
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

dump(mlb, open('/content/gdrive/My Drive/P5/mlb_transformer.pkl', 'wb'))

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

# Load BERT from the Tensorflow Hub
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def feature_USE_fct(sentences, b_size):
    batch_size = b_size

    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        feat = embed(sentences[idx:idx + batch_size])
        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))
    return features


b_size = 7
X_train_use = feature_USE_fct(X_train, b_size)
X_test_use = feature_USE_fct(X_test, b_size)

########################################
#########PIPELINE & FITTING#############
########################################

#print('Training...')
t0=time()
final_model = Pipeline([('model', OneVsRestClassifier(LogisticRegression(C=1.0, penalty='l2', solver='sag', random_state=1)))])
final_model.fit(X_train_use, y_train)

print("Training finished!")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")

########################################
##############PREDICTION ###############
########################################

print("Prediction...")
t0 = time()

y_pred = final_model.predict(X_test_use)
print("Prediction finished!")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")


######################################
############MODEL SAVING##############
######################################

print("Saving model...")
t0 = time()

joblib.dump(final_model, "/content/gdrive/My Drive/P5/model_pipeline.pkl")
print("Model saved!")
print("done in %0.3fs." % (time() - t0))
print("-----------------------------")

######################################
###########MODEL TRIAL################
######################################

print("Model loading...")

final_model = joblib.load("/content/gdrive/My Drive/P5/model_pipeline.pkl")
transformer = joblib.load("/content/gdrive/My Drive/P5/mlb_transformer.pkl")

print("Model loaded!")
t0 = time()
my_prediction = final_model.predict(X_test_use)
my_prediction = mlb.inverse_transform(my_prediction)
print("done in %0.3fs." % (time() - t0))

my_prediction = [x for x in my_prediction if x != ()]
print("My tags", list(set(my_prediction)))
