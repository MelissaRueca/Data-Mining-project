import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load processed data
train_df = pd.read_csv("train_reviews.csv")
test_df  = pd.read_csv("test_reviews.csv")
y_train = train_df["label"]
y_test  = test_df["label"]

# Load vectorizers
with open("tfidf_uni.pkl", "rb") as f:
    tfidf_uni = pickle.load(f)

with open("tfidf_bi.pkl", "rb") as f:
    tfidf_bi = pickle.load(f)

# Transform data
X_train_uni = tfidf_uni.transform(train_df["clean_text"])
X_test_uni  = tfidf_uni.transform(test_df["clean_text"])

X_train_bi = tfidf_bi.transform(train_df["clean_text"])
X_test_bi  = tfidf_bi.transform(test_df["clean_text"])


# UNIGRAMS 
print(" Multinomial Naive Bayes - Unigrams")

pipe_uni = Pipeline([
    ('select', SelectKBest(score_func=chi2)),
    ('nb', MultinomialNB())
])

param_grid_uni = {
    'select__k': [500, 1000, 1500, 2000, 2500],
    'nb__alpha': [0.01, 0.1, 0.5, 1.0, 5.0]
}

grid_uni = GridSearchCV(pipe_uni, param_grid_uni, cv=10, n_jobs=-1)
grid_uni.fit(X_train_uni, y_train)

print("Best parameters:", grid_uni.best_params_)

y_pred_uni = grid_uni.predict(X_test_uni)

print("Accuracy:", accuracy_score(y_test, y_pred_uni))
print("Precision:", precision_score(y_test, y_pred_uni, pos_label='deceptive'))
print("Recall:", recall_score(y_test, y_pred_uni, pos_label='deceptive'))
print("F1-score:", f1_score(y_test, y_pred_uni,pos_label='deceptive'))
print("Classification Report:", classification_report(y_test, y_pred_uni))

# Top 5 features - Unigrams

# Get trained components
nb_model = grid_uni.best_estimator_.named_steps['nb']
selector = grid_uni.best_estimator_.named_steps['select']

# Feature names after selection
feature_names = np.array(tfidf_uni.get_feature_names_out())[selector.get_support()]

# Log probabilities of features for each class
feature_log_probs = nb_model.feature_log_prob_  # shape (n_classes, n_features)

# Difference in log probabilities between classes
log_prob_diff = feature_log_probs[1] - feature_log_probs[0]

# Top 5 features for deceptive (class 1) and truthful (class 0) reviews
top_k_deceptive = np.argsort(log_prob_diff)[-5:][::-1]
top_k_truthful = np.argsort(-log_prob_diff)[-5:][::-1]

print("Top 5 features for deceptive reviews:", feature_names[top_k_deceptive])
print("Top 5 features for truthful reviews:", feature_names[top_k_truthful])


#  UNIGRAMS AND BIGRAMS 
print("Multinomial Naive Bayes - Unigrams and Bigrams ")

pipe_bi = Pipeline([
    ('select', SelectKBest(score_func=chi2)),
    ('nb', MultinomialNB())
])

param_grid_bi = {
    'select__k': [1000, 2000, 3000, 5000, 8000],
    'nb__alpha': [0.01, 0.1, 0.5, 1.0, 5.0]
}

grid_bi = GridSearchCV(pipe_bi, param_grid_bi, cv=10, n_jobs=-1)
grid_bi.fit(X_train_bi, y_train)

print("Best parameters:", grid_bi.best_params_)

y_pred_bi = grid_bi.predict(X_test_bi)

print("Accuracy:", accuracy_score(y_test, y_pred_bi))
print("Precision:", precision_score(y_test, y_pred_bi, pos_label='deceptive'))
print("Recall:", recall_score(y_test, y_pred_bi,pos_label='deceptive'))
print("F1-score:", f1_score(y_test, y_pred_bi, pos_label='deceptive'))
print("Classification Report:", classification_report(y_test, y_pred_bi))

# Top 5 features - Unigrams + Bigrams

# Get trained components
nb_model_bi = grid_bi.best_estimator_.named_steps['nb']
selector_bi = grid_bi.best_estimator_.named_steps['select']

# Feature names after selection
feature_names_bi = np.array(tfidf_bi.get_feature_names_out())[selector_bi.get_support()]

# Log probabilities of features for each class
feature_log_probs_bi = nb_model_bi.feature_log_prob_  

# Difference in log probabilities between classes
log_prob_diff_bi = feature_log_probs_bi[1] - feature_log_probs_bi[0]

# Top 5 features for deceptive (class 1) and truthful (class 0) reviews
top_k_deceptive_bi = np.argsort(log_prob_diff_bi)[-5:][::-1]
top_k_truthful_bi = np.argsort(-log_prob_diff_bi)[-5:][::-1]

print("Top 5 features for deceptive reviews:", feature_names_bi[top_k_deceptive_bi])
print("Top 5 features for truthful reviews:", feature_names_bi[top_k_truthful_bi])

# Save predictions
with open("pred_naive_bayes_unigrams.pkl", "wb") as f:
    pickle.dump(y_pred_uni, f)
    
with open("pred_naive_bayes_bigrams.pkl", "wb") as f:
    pickle.dump(y_pred_bi, f)