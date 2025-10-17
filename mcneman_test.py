import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(csv1, csv2):
    """
    Performs McNemar's test between two classifier result CSVs.
    Both CSVs must have columns: 'text', 'true_label', and 'predicted_label'.
    """

    #Load CSVs
    a = pd.read_csv(csv1)
    b = pd.read_csv(csv2)

    #Merge on text + true_label (only matching rows)
    merged = pd.merge(a, b, on=["text", "true_label"], suffixes=("_a", "_b"))

    #Check correctness for each model
    merged["a_correct"] = merged["predicted_label_a"] == merged["true_label"]
    merged["b_correct"] = merged["predicted_label_b"] == merged["true_label"]

    #Discordant pairs
    n01 = ((~merged["a_correct"]) & (merged["b_correct"])).sum()  # A wrong, B right
    n10 = ((merged["a_correct"]) & (~merged["b_correct"])).sum()  # A right, B wrong

    #McNemar’s test
    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=False, correction=True)

    #Results
    print("\nMcNemar's Test Results:")
    print(f"n01 (A wrong, B right): {n01}")
    print(f"n10 (A right, B wrong): {n10}")
    print(f"Chi-square statistic: {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.4f}")

    if result.pvalue < 0.05:
        print("\n✅ Significant difference (p < 0.05).")
    else:
        print("\n❌ No significant difference (p ≥ 0.05).")

#Example usage:
#mcnemar_test("predictions_Logistic_Regression_Unigram.csv", "predictions_Classification_Tree_UnigramF.csv")





from itertools import combinations

def mcnemar_all(file_list):
    """
    Run McNemar's test for every unique pair of CSVs in file_list.
    Uses the existing mcnemar_test(csv1, csv2) function you already have.
    """
    for csv1, csv2 in combinations(file_list, 2):
        print("\n" + "="*70)
        print(f"Comparing: {csv1}  vs  {csv2}")
        print("="*70)
        mcnemar_test(csv1, csv2)

files = [

    #"predictions_MNB_unigram.csv",
    "predictions_MNB_unigramandbigram.csv", 
    #"predictions_LR_unigram.csv",
    "predictions_LR_unigramandbigram.csv",
    #"predictions_CT_unigram.csv",
    "predictions_CT_unigramandbigram.csv", 
    #"predictions_RF_unigram.csv",
    "predictions_RF_unigramandbigram.csv",
    #"predictions_GB_unigram.csv",
    "predictions_GB_unigramandbigram.csv"
]
#mcnemar_all(files)