import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations

def mcnemar_pvalue(csv1, csv2):
    a = pd.read_csv(csv1)
    b = pd.read_csv(csv2)
    merged = pd.merge(a, b, on=["text", "true_label"], suffixes=("_a", "_b"))

    #correctness flags
    a_correct = merged["predicted_label_a"] == merged["true_label"]
    b_correct = merged["predicted_label_b"] == merged["true_label"]

    #discordant pairs
    n01 = ((~a_correct) & (b_correct)).sum()  # A wrong, B right
    n10 = ((a_correct) & (~b_correct)).sum()  # A right, B wrong

    #contingency table with only discordant counts
    table = [[0, n01], [n10, 0]]
    res = mcnemar(table, exact=False, correction=True)
    return float(res.pvalue)

def mcnemar_heatmap(file_list, display_names=None, figsize=(10, 8), annotate=True):
    """
    file_list: list of CSV paths
    display_names: optional list of labels (same length/order as file_list)
    """
    n = len(file_list)
    labels = display_names if display_names else file_list

    #p-value matrix
    P = np.ones((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        p = mcnemar_pvalue(file_list[i], file_list[j])
        P[i, j] = P[j, i] = p

    #plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(P, aspect='auto')  # default colormap

    #ticks & labels
    ax.set_xticks(range(n), labels=labels, rotation=45, ha='right')
    ax.set_yticks(range(n), labels=labels)

    #colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("McNemar p-value")

    #annotations
    if annotate:
        for i in range(n):
            for j in range(n):
                txt = f"{P[i, j]:.2g}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=8)

    ax.set_title("P-value Matrix for Model & N-gram Comparisons")
    plt.tight_layout()
    plt.show()

#Example: nice short labels to match your screenshot
labels = [
    "Multinomial Naive Bayes (both)",
    "Logistic Regression (both)",

    "Classification Tree (both)",
    "Random Forest (both)",
    "Gradient Boosting (both)"
    #add more if you include them in `files`
]

#Make sure labels length matches files length or pass None to use filenames

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

if __name__ == "__main__":
    mcnemar_heatmap(files, display_names=labels)
#mcnemar_heatmap(files, display_names=labels)
