# knn.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class KNN:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=1)
    def fit(self, x, y):
        self.knn.fit(x, y)
    def predict(self, x):
        return self.knn.predict(x)
    def eval(self, x, y_hat, metric=accuracy_score):
        y_pred = self.predict(x)
        return metric(y_pred, y_hat)
    def fit_and_eval(self, train_x, train_y, test_x, test_y, metric=accuracy_score):
        self.fit(train_x, train_y)
        return self.eval(test_x, test_y, metric)




def KnnClassifier(train_df, test_df, sample_size, max_features=5000, n_neighbors=1):

    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(train_df["text"])

    X_train = vectorizer.transform(train_df.iloc[:sample_size]["text"]).toarray()
    y_train = train_df.iloc[:sample_size]["label_id"].values
    X_test = vectorizer.transform(test_df["text"]).toarray()
    y_test = test_df["label_id"].values
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["contradiction", "entailment", "neutral"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}
