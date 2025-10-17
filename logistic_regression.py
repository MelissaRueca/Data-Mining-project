import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pr import prepare_data

def plot_confusion_matrix(y_true, y_pred, labels=("truthful", "deceptive"), model_name="Logistic Regression", ngram_label="Unigram"):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name} - {ngram_label}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def logistic_regression_pipeline(train_text, train_y, test_text, test_y, ngram_range=(1,1), name="Unigrams"):
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            ngram_range=ngram_range,
            max_features=1000,
            lowercase=False
        )),
        ("lr", LogisticRegressionCV(
            Cs=np.logspace(-1, 1, 5),
            cv=inner_cv,
            penalty="l1",
            solver="liblinear",
            scoring="accuracy",
            max_iter=5000,
            n_jobs=-1,
            refit=True,
            random_state=42
        ))
    ])

    pipe.fit(train_text, train_y)

    chosen_C = pipe.named_steps["lr"].C_
    print("Best C value found:", chosen_C)

    y_pred = pipe.predict(test_text)
    acc = accuracy_score(test_y, y_pred)
    print("Test Accuracy:", round(acc, 4))
    print("\nClassification Report:\n", classification_report(test_y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(test_y, y_pred))

    pd.DataFrame({
        "text": test_text,
        "true_label": test_y,
        "predicted_label": y_pred
    }).to_csv(f"predictions_{name.replace(' ', '_')}.csv", index=False)

    ngram_label = "Unigram" if ngram_range == (1,1) else "Unigram+Bigram"
    plot_confusion_matrix(test_y, y_pred,labels=("truthful", "deceptive"),
                           model_name="Logistic Regression",ngram_label=ngram_label)
    return acc, chosen_C, pipe

def show_top_terms(pipe, top_n=5):
    vectorizer = pipe.named_steps["tfidf"]
    model = pipe.named_steps["lr"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]

    print(f"\nTop {top_n} terms pointing to FAKE (deceptive) reviews:")
    for i in top_fake_idx:
        print(f"{feature_names[i]:<20} {coefs[i]:.4f}")

    print(f"\nTop {top_n} terms pointing to GENUINE (truthful) reviews:")
    for i in top_real_idx:
        print(f"{feature_names[i]:<20} {coefs[i]:.4f}")

    return pd.DataFrame({"term": feature_names, "coef": coefs}).sort_values("coef", ascending=False)

if __name__ == "__main__":
    # Load and merge datasets
    train_df, test_df, y_train, y_test = prepare_data()
    
    # Run the function for unigrams
    acc_uni, C_uni, pipe_uni = logistic_regression_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,1),
        name="LR_unigrams"
    )

    # Run the function for unigrams + bigrams
    acc_bi, C_bi, pipe_bi = logistic_regression_pipeline(
        train_df["clean_text"], y_train,
        test_df["clean_text"], y_test,
        ngram_range=(1,2),
        name="LR_unigramsandbigrams"
    )

    print("\n--- Final Comparison ---")
    print(f"Unigram accuracy        : {acc_uni:.4f}")
    print(f"Unigram + Bigram accuracy: {acc_bi:.4f}")

    show_top_terms(pipe_uni, top_n=5)
    show_top_terms(pipe_bi, top_n=5)