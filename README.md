# Data-Mining-project

## How to run the code
To execute the code and reproduce the results, run 
```bash
python main.py
```
This script:
- Preprocesses the dataset  
- Trains and evaluates all models (Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)  
- Saves prediction files (`predictions_*.csv`)  
- Runs McNemar statistical tests and generates the heatmap comparison  

You can also run individual model scripts, for example:
```bash
python logistic_regression.py
python random_forest.py
python gradient_boosting.py
```

## Scripts that produce key results

| Purpose | Script(s) |
|----------|------------|
| **Data preprocessing** | `pr.py` |
| **Model training and evaluation** | `Multinomial_Naive_Bayes.py`, `logistic_regression.py`, `classification_tree.py`, `random_forest.py`, `gradient_boosting.py` |
| **Main experimental results (performance metrics)** | `main.py` |
| **Statistical comparison of models (McNemar tests, heatmap)** | `mcneman_test.py`, `mcneman_heatmap.py` |
