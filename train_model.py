import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

DATA_PATH = os.path.join("data", "dataset.csv")

data = pd.read_csv(DATA_PATH, engine="python")

data = data.dropna()

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_vec, y)

if not os.path.exists("model"):
    os.makedirs("model")

joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model trained successfully")
