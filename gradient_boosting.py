import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, make_scorer
)
import joblib
from pr import prepare_data

def plot_confusion_matrix(y_true, y_pred, labels=("truthful", "deceptive"), model_name="Gradient Boosting", ngram_label="Unigram"):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name} - {ngram_label}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def show_gb_feature_importance(pipe, top_n=5):
    vectorizer = pipe.named_steps["tfidf"]
    gb = pipe.named_steps["gb"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = gb.feature_importances_
    idx = np.argsort(importances)[-top_n:][::-1]

    print(f"\nTop {top_n} features by importance (Gradient Boosting):")
    for i in idx:
        print(f"{feature_names[i]:<25} {importances[i]:.4f}")

def gradient_boosting_pipeline(train_text, train_y, test_text, test_y, ngram_range=(1,1), name="unigrams"):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_y)
    y_test_enc = le.transform(test_y)

    inner_cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            ngram_range=ngram_range,
            lowercase=False
        )),
        ("gb", GradientBoostingClassifier(random_state=42))
    ])

    param_dist = {
        "tfidf__max_features": [2000, 5000, 8000] if ngram_range == (1,1) else [5000, 8000, 12000],
        "gb__n_estimators": [200, 300],
        "gb__max_depth": [3, 5],
        "gb__learning_rate": [0.05, 0.1],
        "gb__subsample": [0.7, 0.85],
        "gb__max_features": ["sqrt", "log2"]
    }

    scorer = make_scorer(f1_score)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring=scorer,
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    search.fit(train_text, y_train_enc)
    best_pipe = search.best_estimator_
    best_params = search.best_params_
    print("Best params:", best_params)

    y_pred_enc = best_pipe.predict(test_text)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred, pos_label='deceptive')
    rec = recall_score(test_y, y_pred, pos_label='deceptive')
    f1 = f1_score(test_y, y_pred, pos_label='deceptive')

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(test_y, y_pred))

    pd.DataFrame({
    "text": test_text,
    "true_label": test_y,
    "predicted_label": y_pred
    }).to_csv(f"predictions_{name}.csv", index=False)

    joblib.dump(best_pipe, f"gb_{name}.pkl")

    ngram_label = "Unigram" if ngram_range == (1,1) else "Unigram+Bigram"
    plot_confusion_matrix(test_y, y_pred,
                          labels=("truthful", "deceptive"),
                          model_name="Gradient Boosting",
                          ngram_label=ngram_label)

    return acc, best_params, best_pipe

if __name__ == "__main__":
    train_df, test_df, y_train, y_test = prepare_data()

    acc_gb_uni, params_gb_uni, pipe_gb_uni = gradient_boosting_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,1),
        name="unigrams"  
    )

    acc_gb_bi, params_gb_bi, pipe_gb_bi = gradient_boosting_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,2),
        name="unigramsandbigrams"  
    )

    show_gb_feature_importance(pipe_gb_uni, top_n=5)
    show_gb_feature_importance(pipe_gb_bi, top_n=5)

    print("\n--- Gradient Boosting: Final Comparison ---")
    print(f"Unigram accuracy         : {acc_gb_uni:.4f}")
    print(f"Unigram + Bigram accuracy: {acc_gb_bi:.4f}")

