from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class RF:
    def __init__(self):
        self.rf = RandomForestClassifier(n_jobs=-1, max_samples=.8, n_estimators=50)
    def fit(self, x, y):
        self.rf.fit(x, y)
    def predict(self, x):
        return self.rf.predict(x)
    def eval(self, x, y_hat, metric=accuracy_score):
        y_pred = self.predict(x)
        return metric(y_pred, y_hat)
    def fit_and_eval(self, train_x, train_y, test_x, test_y, metric=accuracy_score):
        self.fit(train_x, train_y)
        return self.eval(test_x, test_y, metric)


