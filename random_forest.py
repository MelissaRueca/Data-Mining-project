import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer

from pr import prepare_data

def plot_confusion_matrix(y_true, y_pred, labels=("truthful", "deceptive"), model_name="Random Forest", ngram_label="Unigram"):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name} - {ngram_label}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def random_forest_pipeline(train_text, train_y, test_text, test_y, ngram_range=(1,1), name="Unigrams"):
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            ngram_range=ngram_range,
            max_features=1000,
            lowercase=False
        )),
        ("rf", RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            bootstrap=True
        ))
    ])

    param_grid = {
        "rf__n_estimators": [300, 400, 500],
        "rf__max_depth": [None, 20, 30],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2],
        "rf__max_features": ["sqrt", "log2"]
    }

    scorer = make_scorer(f1_score, pos_label="deceptive")

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scorer,
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    grid.fit(train_text, train_y)

    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    print("Best params:", best_params)
    if hasattr(best_pipe.named_steps["rf"], "oob_score_"):
        print("OOB Score:", round(best_pipe.named_steps["rf"].oob_score_, 4))

    y_pred = best_pipe.predict(test_text)
    acc = accuracy_score(test_y, y_pred)
    print("Test Accuracy:", round(acc, 4))
    print("\nClassification Report:\n", classification_report(test_y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(test_y, y_pred))

    # Save true labels, predicted labels, and text to CSV
    pd.DataFrame({
        "text": test_text,
        "true_label": test_y,
        "predicted_label": y_pred
    }).to_csv(f"predictions_{name.replace(' ', '_')}.csv", index=False)

    ngram_label = "Unigram" if ngram_range == (1,1) else "Unigram+Bigram"
    plot_confusion_matrix(test_y, y_pred,
                          labels=("truthful", "deceptive"),
                          model_name="Random Forest",
                          ngram_label=ngram_label)

    return acc, best_params, best_pipe

def show_rf_feature_importance(pipe, top_n=15):
    vectorizer = pipe.named_steps["tfidf"]
    rf = pipe.named_steps["rf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-top_n:][::-1]
    print(f"\nTop {top_n} features by importance (Random Forest):")
    for i in idx:
        print(f"{feature_names[i]:<25} {importances[i]:.4f}")

if __name__ == "__main__":
    # Load data
    train_df, test_df, y_train, y_test = prepare_data()

    # Run Random Forest with Unigrams 
    acc_rf_uni, params_rf_uni, pipe_rf_uni = random_forest_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,1),
        name="RF_unigrams"
    )

    # Run Random Forest with Unigrams + Bigrams 
    acc_rf_bi, params_rf_bi, pipe_rf_bi = random_forest_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,2),
        name="RF_unigramsandbigrams"
    )

    show_rf_feature_importance(pipe_rf_uni, top_n=5)
    show_rf_feature_importance(pipe_rf_bi, top_n=5)

    print("\n--- Random Forest: Final Comparison ---")
    print(f"Unigram accuracy         : {acc_rf_uni:.4f}")
    print(f"Unigram + Bigram accuracy: {acc_rf_bi:.4f}")
