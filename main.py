from pr import prepare_data
from Multinomial_Naive_Bayes import run_mnb_pipeline
from classification_tree import decision_tree_pipeline
from gradient_boosting import gradient_boosting_pipeline
from logistic_regression import logistic_regression_pipeline, show_top_terms
from random_forest import random_forest_pipeline
from mcneman_test import mcnemar_all
from mcneman_heatmap import mcnemar_heatmap

if __name__ == "__main__":
    train_df, test_df, y_train, y_test = prepare_data()

    models = {
        "Multinomial Naive Bayes": (run_mnb_pipeline, None),
        "Classification Tree": (decision_tree_pipeline, None),
        "Gradient Boosting": (gradient_boosting_pipeline, None),
        "Logistic Regression": (logistic_regression_pipeline, show_top_terms),
        "Random Forest": (random_forest_pipeline, None)
    }

    ngram_ranges = {"Unigram": (1,1), "Unigram+Bigram": (1,2)}
    short_names = {"Multinomial Naive Bayes":"MNB", "Classification Tree":"CT", "Gradient Boosting":"GB",
                   "Logistic Regression":"LR", "Random Forest":"RF"}

    prediction_files = []

    for model_name, (func, show_func) in models.items():
        print(f"\n--- {model_name} ---")
        if model_name == "Multinomial Naive Bayes":
            run_mnb_pipeline(train_df, y_train, test_df, y_test)
            prediction_files.extend([
                "predictions_MNB_unigram.csv",
                "predictions_MNB_unigramandbigram.csv"
            ])
        else:
            for ngram_label, ngram_range in ngram_ranges.items():
                print(f"--- {ngram_label} ---")
                acc, params, pipe = func(
                    train_df["clean_text"], y_train,
                    test_df["clean_text"], y_test,
                    ngram_range=ngram_range,
                    name=f"{short_names[model_name]}_{ngram_label.lower().replace('+','and')}"
                )
                file_name = f"predictions_{short_names[model_name]}_{'unigram' if ngram_label=='Unigram' else 'unigramandbigram'}.csv"
                prediction_files.append(file_name)
                if show_func: show_func(pipe, top_n=5)

    print("\n--- McNemar tests ---")
    display_names = [f"{m} ({n})" for m in models.keys() for n in ngram_ranges.keys()]
    mcnemar_all(prediction_files)
    mcnemar_heatmap(prediction_files, display_names=display_names)
