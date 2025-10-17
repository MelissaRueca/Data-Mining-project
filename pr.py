import os
import random
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

random.seed(42)
np.random.seed(42)

#Keep negations
stop_words = set(stopwords.words('english')) 
stop_words.discard('not') 
stop_words.discard('no')
stop_words.discard("n't") 

lemmatizer = WordNetLemmatizer()

def load_reviews(path, label):
    """
    Load reviews from fold directories and assign labels.
    """
    data = []
    for fold in range(1,6):
        folder = path + f"/fold{fold}"
        for file in sorted(os.listdir(folder)):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read().strip()
                data.append({
                    "text": text,
                    "label": label,
                    "fold": fold
                })
    return pd.DataFrame(data)

def preprocess_text(text):
    """
    Preprocessing Text :
    1. Convert to Lowercase
    2. Removes HTML tags
    3. Removes punctuation and digits
    4. Tokenization
    5. Stopword removal (keeping negations)
    6. Lemmatization
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def prepare_data():
    """
    Loads, preprocesses, and splits the dataset into training and test sets.
    Returns:
        train_df, test_df, y_train, y_test
    """
    truthful_df = load_reviews("negative_polarity/truthful_from_Web", "truthful") 
    deceptive_df = load_reviews("negative_polarity/deceptive_from_MTurk", "deceptive") 
    df = pd.concat([truthful_df, deceptive_df], ignore_index=True)

    #Apply preprocessin
    df["clean_text"] = df["text"].apply(preprocess_text)

    #Split folds 1-4 for training and fold 5 for testing
    train_df = df[df["fold"] != 5]
    test_df = df[df["fold"] == 5]

    #Extract labels
    y_train = train_df["label"]
    y_test = test_df["label"]

    return train_df, test_df, y_train, y_test

