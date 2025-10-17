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

def plot_confusion_matrix(y_true, y_pred, labels=("truthful", "deceptive"), model_name="Multinomial Naive Bayes", ngram_label="Unigram"):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name} - {ngram_label}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def run_mnb_pipeline(train_df, y_train, test_df, y_test):
    pipe_uni = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1,1))),
        ('select', SelectKBest(score_func=chi2)),
        ('nb', MultinomialNB())
    ])
    
    param_grid_uni = {
        'tfidf__max_features': [2000, 5000, 8000, 10000],
        'select__k': [500, 1000, 1500, 2000], 
        'nb__alpha': [0.01, 0.1, 0.5, 1.0]
    }
    grid_uni = GridSearchCV(pipe_uni, param_grid_uni, cv=5, n_jobs=-1)
    grid_uni.fit(train_df["clean_text"], y_train)
    y_pred_uni = grid_uni.predict(test_df["clean_text"])

    print("--- Unigrams ---")
    print("Best parameters:", grid_uni.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred_uni))
    print("Precision:", precision_score(y_test, y_pred_uni, pos_label='deceptive'))
    print("Recall:", recall_score(y_test, y_pred_uni, pos_label='deceptive'))
    print("F1-score:", f1_score(y_test, y_pred_uni, pos_label='deceptive'))
    print("Classification Report:\n", classification_report(y_test, y_pred_uni))
    plot_confusion_matrix(y_test, y_pred_uni, ngram_label="Unigram")

    pd.DataFrame({
        "text": test_df["clean_text"],
        "true_label": y_test,
        "predicted_label": y_pred_uni
    }).to_csv("predictions_MNB_unigram.csv", index=False)

    pipe_bi = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1,2))),
        ('select', SelectKBest(score_func=chi2)),
        ('nb', MultinomialNB())
    ])
    param_grid_bi = {
        'tfidf__max_features': [5000, 8000, 10000, 12000],
        'select__k': [1000, 2000, 3000, 5000],
        'nb__alpha': [0.01, 0.1, 0.5, 1.0]
    }
    grid_bi = GridSearchCV(pipe_bi, param_grid_bi, cv=5, n_jobs=-1)
    grid_bi.fit(train_df["clean_text"], y_train)
    y_pred_bi = grid_bi.predict(test_df["clean_text"])

    print("--- Unigrams+Bigrams ---")
    print("Best parameters:", grid_bi.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred_bi))
    print("Precision:", precision_score(y_test, y_pred_bi, pos_label='deceptive'))
    print("Recall:", recall_score(y_test, y_pred_bi, pos_label='deceptive'))
    print("F1-score:", f1_score(y_test, y_pred_bi, pos_label='deceptive'))
    print("Classification Report:\n", classification_report(y_test, y_pred_bi))
    plot_confusion_matrix(y_test, y_pred_bi, ngram_label="Unigram + Bigram")

    pd.DataFrame({
        "text": test_df["clean_text"],
        "true_label": y_test,
        "predicted_label": y_pred_bi
    }).to_csv("predictions_MNB_unigramandbigram.csv", index=False)

    return y_pred_uni, y_pred_bi

if __name__ == "__main__":
    train_df, test_df, y_train, y_test = prepare_data()
    run_mnb_pipeline(train_df, y_train, test_df, y_test)