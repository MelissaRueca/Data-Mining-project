import os
import pandas as pd
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 

# Load reviews
def load_reviews(path, label):
    """
    Load reviews from fold directories and assign labels.
    """
    data = []
    for fold in range(1,6):
        folder = path + f"/fold{fold}"
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read().strip()
                data.append({
                    "text": text,
                    "label": label,
                    "fold": fold
                })
    return pd.DataFrame(data)

# Load and merge datasets 
truthful_df = load_reviews("negative_polarity/truthful_from_Web", "truthful") 
deceptive_df = load_reviews("negative_polarity/deceptive_from_MTurk", "deceptive") 
df = pd.concat([truthful_df, deceptive_df], ignore_index=True) 

# Keep negations
stop_words = set(stopwords.words('english')) 
stop_words.discard('not') 
stop_words.discard('no')
stop_words.discard("n't") 

lemmatizer = WordNetLemmatizer()

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

# Apply preprocessing 
df["clean_text"] = df["text"].apply(preprocess_text)

# Split folds 1-4 for training and fold 5 for testing
train_df = df[df["fold"] != 5]
test_df = df[df["fold"] == 5]

# Extract labels
y_train = train_df["label"]
y_test = test_df["label"]

# Feature extraction

# Unigrams (single words)
tfidf_uni = TfidfVectorizer(
        max_df=0.95,  # exclude terms that appear in >95% of documents
        min_df=2, # exclude terms that appear <2 documents
        ngram_range=(1,1), 
        lowercase=False  
)

X_train_uni = tfidf_uni.fit_transform(train_df["clean_text"])
X_test_uni= tfidf_uni.transform(test_df["clean_text"]) 

# Unigrams + bigrams (single + two-word phrases)
tfidf_bi = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),
        lowercase=False
)

X_train_bi = tfidf_bi.fit_transform(train_df["clean_text"])
X_test_bi = tfidf_bi.transform(test_df["clean_text"])

# Save processed data
df.to_csv("processed_reviews.csv", index=False) 
train_df.to_csv("train_reviews.csv", index=False)
test_df.to_csv("test_reviews.csv", index=False)

# Save vectorizers 
with open("tfidf_uni.pkl", "wb") as f:
    pickle.dump(tfidf_uni, f)
with open("tfidf_bi.pkl", "wb") as f:
    pickle.dump(tfidf_bi, f)

print("Data preprocessing complete.")

