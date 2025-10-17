# Data-Mining-project

# Classification for the Detection of Opinion Spam Data Mining Assignment 2025.
This project compares multiple ML algorithms with statistical validation to identify the most effective approach for deception detection.

## Project Overview
This project implements and evaluates five machine learning classifiers for binary text classification (truthful vs. deceptive reviews):
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Classification Tree**
- **Random Forest**
- **Gradient Boosting**
Each model is evaluated with both unigram and bigram feature representations, with comprehensive statistical comparison using McNemar's test.

### Installation
```bash
pip install -r requirements.txt
```

Additionally, download the NLTK language resources (only required once):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## How to run the code
To execute the code and reproduce the results, run 
```bash
python main.py
```
This script:
- Preprocesses the dataset  
- Trains and evaluates all models (Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Tests both unigrams and bigram feature configurations
- Displays confusion matrices (pop-up windows) for each model
👉 Close each window to continue execution
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
```
