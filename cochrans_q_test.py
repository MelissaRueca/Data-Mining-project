import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from itertools import combinations

def cochrans_q_with_pairwise_mcnemar(csv_files, model_names=None):
    """
    Cochran's Q test for multiple classifiers, followed by pairwise McNemar if significant.
    """
    # Load CSVs
    dfs = [pd.read_csv(f) for f in csv_files]

    # Merge all predictions on 'text'
    merged = dfs[0][['text', 'true_label', 'predicted_label']].copy()
    merged.rename(columns={'predicted_label': 'model_0'}, inplace=True)

    for i, df in enumerate(dfs[1:], start=1):
        merged = pd.merge(merged,
                          df[['text', 'predicted_label']],
                          on='text')
        merged.rename(columns={'predicted_label': f'model_{i}'}, inplace=True)

    # Binary correctness DataFrame
    correct_df = pd.DataFrame()
    n_models = len(dfs)
    for i in range(n_models):
        col_name = f'model_{i}' if not model_names else model_names[i]
        correct_df[col_name] = merged[f'model_{i}'] == merged["true_label"]

    print("✅ Preview of correctness matrix:")
    print(correct_df.head(), "\n")

    # Cochran's Q test
    result = cochrans_q(correct_df)
    Q = result.statistic
    p = result.pvalue

    print("Cochran's Q Test Results:")
    print(f"Q statistic: {Q:.4f}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print("\n✅ Significant difference detected. Performing pairwise McNemar tests...\n")

        # Pairwise McNemar
        pairs = combinations(correct_df.columns, 2)
        for a, b in pairs:
            table = pd.crosstab(correct_df[a], correct_df[b])
            mcnemar_result = mcnemar(table, exact=False, correction=True)
            sig = "✅" if mcnemar_result.pvalue < 0.05 else "❌"
            print(f"{a} vs {b}: chi2={mcnemar_result.statistic:.4f}, p={mcnemar_result.pvalue:.4f} {sig}")
    else:
        print("\n❌ No significant differences detected among models.")

# Example usage
csv_files = [
    "predictions_gb_Uni.csv",
    "predictions_Classification_Tree_Unigram.csv",
    "predictions_MNB_unigrams.csv",
    "predictions_rf_Uni.csv",
    "predictions_Uni.csv"
]
model_names = ["GB Uni", "Tree Uni", "NB", "RF", "LR"]

cochrans_q_with_pairwise_mcnemar(csv_files, model_names)