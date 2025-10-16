import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pr import prepare_data  
import matplotlib.pyplot as plt
import seaborn as sns

#  Load data 
train_df, test_df, y_train, y_test = prepare_data()

# Label encoding
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

def plot_confusion_matrix(y_true, y_pred, labels=("truthful", "deceptive"), model_name="Gradient Boosting", ngram_label="Unigram"):
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
    
#  Feature sets and pipelines
feature_sets = [
    ("unigrams", (1,1)),
    ("unigramsandbigrams", (1,2))
]

all_results = []

for feat_name, ngram_range in feature_sets:

    print(f"\nGradient Boosting ---- {feat_name}")

    
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=ngram_range)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])

    # Parameter grid for RandomizedSearchCV
    param_dist = {
        'tfidf__max_features': [5000, 8000, 12000] if feat_name=="unigramsandbigrams" else [2000, 5000, 8000],
        'gb__n_estimators': [200, 300],
        'gb__max_depth': [3, 5],
        'gb__learning_rate': [0.05, 0.1],
        'gb__subsample': [0.7, 0.85],
        'gb__max_features': ['sqrt', 'log2']
    }

    cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=20,
        cv=cv, scoring='f1', n_jobs=-1, random_state=42
    )

    # Fit on training data
    search.fit(train_df["clean_text"], y_train_enc)

    # Best model and params
    best_model = search.best_estimator_
    best_params = search.best_params_
    print("Best params:", best_params)

    # Evaluate on test data
    y_pred_enc = best_model.predict(test_df["clean_text"])
    metrics = {
        "Accuracy": accuracy_score(y_test_enc, y_pred_enc),
        "Precision": precision_score(y_test_enc, y_pred_enc),
        "Recall": recall_score(y_test_enc, y_pred_enc),
        "F1-score": f1_score(y_test_enc, y_pred_enc)
    }
    print("Metrics:", metrics)
    # Decode predictions back to original labels for plotting
    y_pred_labels = le.inverse_transform(y_pred_enc)
    # Plot confusion matrix (using original string labels)
    plot_confusion_matrix(y_test, y_pred_labels,
                        labels=("truthful", "deceptive"),
                        model_name="Gradient Boosting",
                        ngram_label=feat_name)
    # Save predictions
    pd.DataFrame({
        "text": test_df["clean_text"],
        "true_label": y_test,
        "predicted_label": le.inverse_transform(y_pred_enc)
    }).to_csv(f"predictions_GB_{feat_name}.csv", index=False)

    # Save model
    joblib.dump(best_model, f"gb_{feat_name}.pkl")

    all_results.append({"Model": "Gradient Boosting", "Features": feat_name, **metrics})

# Performance summary
results_df = pd.DataFrame(all_results)
print("\n---Performance summary---")
print(results_df)
results_df.to_csv("gb_results.csv", index=False)
