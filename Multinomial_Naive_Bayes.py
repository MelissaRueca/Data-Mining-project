import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from pr import prepare_data  
import matplotlib.pyplot as plt
import seaborn as sns

#  Load and preprocess data
train_df, test_df, y_train, y_test = prepare_data()

def plot_confusion_matrix(y_true, y_pred, labels=("truthful", "deceptive"), model_name="Multinomial Naive Bayes", ngram_label="Unigram"):
    """
    Plots a confusion matrix for a given model’s predictions.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name} - {ngram_label}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

#  Multinomial Naive Bayes - Unigrams

print("Multinomial Naive Bayes - Unigrams")

pipe_uni = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2,ngram_range=(1,1))),
    ('select', SelectKBest(score_func=chi2)),
    ('nb', MultinomialNB())
])


param_grid_uni = {
    'tfidf__max_features': [2000, 5000, 8000,10000],
    'select__k': [500, 1000, 1500, 2000], 
    'nb__alpha': [0.01, 0.1, 0.5, 1.0]
}

grid_uni = GridSearchCV(pipe_uni, param_grid_uni, cv=10, n_jobs=-1)
grid_uni.fit(train_df["clean_text"], y_train)

print("Best parameters:", grid_uni.best_params_)

y_pred_uni = grid_uni.predict(test_df["clean_text"])

print("Accuracy:", accuracy_score(y_test, y_pred_uni))
print("Precision:", precision_score(y_test, y_pred_uni, pos_label='deceptive'))
print("Recall:", recall_score(y_test, y_pred_uni, pos_label='deceptive'))
print("F1-score:", f1_score(y_test, y_pred_uni,pos_label='deceptive'))
print("Classification Report:\n", classification_report(y_test, y_pred_uni))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_uni))

# Plot confusion matrix for Unigram model
plot_confusion_matrix(y_test, y_pred_uni, 
                      labels=("truthful", "deceptive"),
                      model_name="Multinomial Naive Bayes",
                      ngram_label="Unigram")

# Top 5 features - Unigrams

# Get trained components
tfidf_uni_best = grid_uni.best_estimator_.named_steps['tfidf']
selector_uni = grid_uni.best_estimator_.named_steps['select']
nb_model_uni = grid_uni.best_estimator_.named_steps['nb']

# Feature names after selection 
feature_names_uni = np.array(tfidf_uni_best.get_feature_names_out())[selector_uni.get_support()]

# Log probabilities of features for each class and Difference in log probabilities between classes
log_prob_diff_uni = nb_model_uni.feature_log_prob_[1] - nb_model_uni.feature_log_prob_[0]

# Top 5 features for deceptive (class 1) and truthful (class 0) reviews
top_k_deceptive_uni = np.argsort(log_prob_diff_uni)[-5:][::-1]
top_k_truthful_uni = np.argsort(-log_prob_diff_uni)[-5:][::-1]

print("Top 5 features for deceptive reviews:", feature_names_uni[top_k_deceptive_uni])
print("Top 5 features for truthful reviews:", feature_names_uni[top_k_truthful_uni])

#  Multinomial Naive Bayes - Unigrams + Bigrams

print("Multinomial Naive Bayes - Unigrams + Bigrams")

pipe_bi = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.95,min_df=2,ngram_range=(1,2))),
    ('select', SelectKBest(score_func=chi2)),
    ('nb', MultinomialNB())
])


param_grid_bi = {
    'tfidf__max_features': [5000, 8000, 10000,12000],
    'select__k': [1000, 2000, 3000, 5000],  
    'nb__alpha': [0.01, 0.1, 0.5, 1.0]
}

grid_bi = GridSearchCV(pipe_bi, param_grid_bi, cv=10, n_jobs=-1)
grid_bi.fit(train_df["clean_text"], y_train)

print("Best parameters:", grid_bi.best_params_)

y_pred_bi = grid_bi.predict(test_df["clean_text"])

print("Accuracy:", accuracy_score(y_test, y_pred_bi))
print("Precision:", precision_score(y_test, y_pred_bi, pos_label='deceptive'))
print("Recall:", recall_score(y_test, y_pred_bi,pos_label='deceptive'))
print("F1-score:", f1_score(y_test, y_pred_bi, pos_label='deceptive'))
print("Classification Report:\n", classification_report(y_test, y_pred_bi))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bi))

# Plot confusion matrix for Unigram + Bigram model
plot_confusion_matrix(y_test, y_pred_bi, 
                      labels=("truthful", "deceptive"),
                      model_name="Multinomial Naive Bayes",
                      ngram_label="Unigram + Bigram")

# Top 5 features - Unigrams + Bigrams

# Get trained components
tfidf_bi_best = grid_bi.best_estimator_.named_steps['tfidf']
selector_bi = grid_bi.best_estimator_.named_steps['select']
nb_model_bi = grid_bi.best_estimator_.named_steps['nb']

# Feature names after selection 
feature_names_bi = np.array(tfidf_bi_best.get_feature_names_out())[selector_bi.get_support()]

# Log probabilities of features for each class and Difference in log probabilities between classes
log_prob_diff_bi = nb_model_bi.feature_log_prob_[1] - nb_model_bi.feature_log_prob_[0]

# Top 5 features for deceptive (class 1) and truthful (class 0) reviews
top_k_deceptive_bi = np.argsort(log_prob_diff_bi)[-5:][::-1]
top_k_truthful_bi = np.argsort(-log_prob_diff_bi)[-5:][::-1]

print("Top 5 features for deceptive reviews:", feature_names_bi[top_k_deceptive_bi])
print("Top 5 features for truthful reviews:", feature_names_bi[top_k_truthful_bi])


#  Save predictions

pd.DataFrame({
    "text": test_df["clean_text"],
    "true_label": y_test,
    "predicted_label": y_pred_uni
}).to_csv("predictions_MNB_unigrams.csv", index=False)

pd.DataFrame({
    "text": test_df["clean_text"],
    "true_label": y_test,
    "predicted_label": y_pred_bi
}).to_csv("predictions_MNB_unigramsandbigrams.csv", index=False)







