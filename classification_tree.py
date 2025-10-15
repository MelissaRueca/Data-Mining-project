import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer

from pr import prepare_data

def plot_confusion_matrix(y_true, y_pred, labels=("truthful", "deceptive"), model_name="Decision Tree", ngram_label="Unigram"):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name} - {ngram_label}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def decision_tree_pipeline(train_text, train_y, test_text, test_y, ngram_range=(1,1), name="Unigrams"):
    print(f"\n=== Decision Tree :: {name} (ngrams={ngram_range}) ===")

    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            ngram_range=ngram_range,
            max_features=1000,
            lowercase=False
        )),
        ("tree", DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        "tree__ccp_alpha": np.linspace(0.0, 0.05, 20),
        "tree__max_depth": [None, 10, 20, 30]
    }

    scorer = make_scorer(f1_score, pos_label="deceptive")

    grid = GridSearchCV(
        pipe, param_grid=param_grid,
        scoring=scorer, cv=inner_cv, n_jobs=-1, refit=True, verbose=0
    )

    grid.fit(train_text, train_y)

    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    best_alpha = best_params.get("tree__ccp_alpha", None)
    print("Best params:", best_params)
    if best_alpha is not None:
        print("Best ccp_alpha (CP):", best_alpha)

    y_pred = best_pipe.predict(test_text)
    acc = accuracy_score(test_y, y_pred)
    print("Test Accuracy:", round(acc, 4))
    print("\nClassification Report:\n", classification_report(test_y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(test_y, y_pred))

    # ...existing code...
    # Save true labels, predicted labels, and text to CSV
    pd.DataFrame({
        "text": test_text,
        "true_label": test_y,
        "predicted_label": y_pred
    }).to_csv(f"predictions_{name.replace(' ', '_')}.csv", index=False)
# ...existing code...

    ngram_label = "Unigram" if ngram_range == (1,1) else "Unigram+Bigram"
    plot_confusion_matrix(test_y, y_pred,
                          labels=("truthful", "deceptive"),
                          model_name="Decision Tree",
                          ngram_label=ngram_label)

    return acc, best_params, best_pipe

def show_tree_feature_importance(pipe, top_n=15):
    vectorizer = pipe.named_steps["tfidf"]
    tree = pipe.named_steps["tree"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = tree.feature_importances_
    idx = np.argsort(importances)[-top_n:][::-1]
    print(f"\nTop {top_n} features by importance (Decision Tree):")
    for i in idx:
        print(f"{feature_names[i]:<25} {importances[i]:.4f}")

if __name__ == "__main__":
    # Load data
    train_df, test_df, y_train, y_test = prepare_data()

    # Run Decision Tree with Unigrams 
    acc_tree_uni, params_tree_uni, pipe_tree_uni = decision_tree_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,1),
        name="CT_unigrams"
    )

    #Run Decision Tree with Unigrams + Bigrams 
    acc_tree_bi, params_tree_bi, pipe_tree_bi = decision_tree_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,2),
        name="CT_unigramsandbigrams"
    )

    print("\n--- Decision Tree: Final Comparison ---")
    print(f"Unigram accuracy         : {acc_tree_uni:.4f}")
    print(f"Unigram + Bigram accuracy: {acc_tree_bi:.4f}")

    show_tree_feature_importance(pipe_tree_uni, top_n=5)
    show_tree_feature_importance(pipe_tree_bi, top_n=5)