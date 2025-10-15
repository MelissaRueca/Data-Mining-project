import joblib
import pandas as pd
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
model = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
param_dist = {
    'n_estimators': [300, 400, 500],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

feature_sets = [
    ("unigrams", X_train_uni, X_test_uni, tfidf_uni),
    ("unigramsandbigrams", X_train_bi, X_test_bi, tfidf_bi)
]

#Hyperparameter search function with cross validation
def search_best_model(model, param_dist, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
        "F1-score": f1_score(y_test, y_pred),
        "OOB_Score": model.oob_score_
    }

#Training and evaluation
all_results = []
for feat_name, X_train, X_test, vectorizer in feature_sets:
    best_model, best_params = search_best_model(model, param_dist, X_train, y_train)
    best_model.fit(X_train, y_train)
    metrics = evaluate_model(best_model, X_test, y_test)

    print("Random Forest --- ", feat_name)
    print("Best params:", best_params)
    print("Metrics:", metrics)

    y_pred = best_model.predict(X_test)
    test_text = test_df["clean_text"].values
    pd.DataFrame({
        "text": test_text,
        "true_label": le.inverse_transform(y_test),
        "predicted_label": le.inverse_transform(y_pred)
    }).to_csv(f"predictions_RF_{feat_name.replace(' ', '_')}.csv", index=False)
    
    model_filename = f"rf_{feat_name}.pkl"
    joblib.dump(best_model, model_filename)
    
    all_results.append({"Model": "Random Forest", "Features": feat_name, **metrics})

results_df = pd.DataFrame(all_results)
print("\n---Performance summary---")
print(results_df)
results_df.to_csv("rf_results.csv", index=False)