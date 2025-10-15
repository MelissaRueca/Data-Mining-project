import joblib
import pandas as pd
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import numpy as np
import random

random.seed(42)
np.random.seed(42)

#Load data and vectorizers
train_df = pd.read_csv("train_reviews.csv")
test_df = pd.read_csv("test_reviews.csv")

with open("tfidf_uni.pkl", "rb") as f:
    tfidf_uni = pickle.load(f)
with open("tfidf_bi.pkl", "rb") as f:
    tfidf_bi = pickle.load(f)

#Label encoding
le = LabelEncoder()
y_train = le.fit_transform(train_df["label"])
y_test = le.transform(test_df["label"])

#TF-IDF transformations
X_train_uni = tfidf_uni.transform(train_df["clean_text"].values)
X_test_uni  = tfidf_uni.transform(test_df["clean_text"].values)
X_train_bi  = tfidf_bi.transform(train_df["clean_text"].values)
X_test_bi   = tfidf_bi.transform(test_df["clean_text"].values)

#Model and hyperparameters
model = GradientBoostingClassifier(random_state=42)
param_dist = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 0.85],
    'max_features': ['sqrt', 'log2']
}

feature_sets = [
    ("Uni", X_train_uni, X_test_uni, tfidf_uni),
    ("Uni+Bi", X_train_bi, X_test_bi, tfidf_bi)
]

#Hyperparameter search function with cross validation
def search_best_model(model, param_dist, X_train, y_train):
    cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20,
                                cv=cv, scoring="f1", n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

#Model performance evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

#Training and evaluation
all_results = []
for feat_name, X_train, X_test, vectorizer in feature_sets:
    best_model, best_params = search_best_model(model, param_dist, X_train, y_train)
    best_model.fit(X_train, y_train)
    metrics = evaluate_model(best_model, X_test, y_test)
    
    print("Gradient Boosting ---- ", feat_name)
    print("Best params:", best_params)
    print("Metrics:", metrics)
    
    model_filename = f"gb_{feat_name}.pkl"
    joblib.dump(best_model, model_filename)
    
    all_results.append({"Model": "Gradient Boosting", "Features": feat_name, **metrics})

results_df = pd.DataFrame(all_results)
print("\n---Performance summary---")
print(results_df)
results_df.to_csv("gb_results.csv", index=False)
