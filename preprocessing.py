import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 

#Load Data - read reviews and assign a label 
def load_reviews(base_dir, label):
    data = []
    for fold in range(1, 6):  # 5 folds
        fold_dir = os.path.join(base_dir, f"fold{fold}") #creates path for each fold directory e.g. truthful_from_Web/fold1
        for filename in os.listdir(fold_dir): # for every file in the fold 
            if filename.endswith(".txt"): #checks if file .txt 
                with open(os.path.join(fold_dir, filename), "r", encoding="utf-8") as f: #opens file to read it with utf-8 encoding
                    text = f.read().strip() # read all the content of file and removes whitespace from start and ending
                    data.append({"review_text": text, "label": label, "fold": fold}) # add dictionary in data list with three columns: the text , the label and which fold number
    return pd.DataFrame(data) 

truthful_df = load_reviews("truthful_from_Web", "truthful") # loads truthful reviews and assign truthful label
deceptive_df = load_reviews("deceptive_from_MTurk", "deceptive") # loads deceptive reviews and assign deceptive label
df = pd.concat([truthful_df, deceptive_df], ignore_index=True) # combines truthful and deceptive in one dataframe

# Configure stopwords (keep negations)
stop_words = set(stopwords.words('english')) #creates set with all english stopwords
stop_words.discard('not') # important:  exclude negative words from stopwords because they are important for analysis
stop_words.discard('no')
stop_words.discard("n't")  # Also keep word with not e.g. can't , don't

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
    # 1. Lowercase
    text = text.lower()
    # 2. removes HTML tags (if exists)
    text = re.sub(r'<.*?>', '', text)
    # 3. Removes numbers and punctuation
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # 4. Tokenization - split text to tokens
    tokens = nltk.word_tokenize(text)
    # 5. Removing tokens that are stopwords + Lemmatization in every word 
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 6. Rejoin tokens into a string with spaces 
    return " ".join(tokens)

#applying preprocessing function to all reviews and save new clean text to a new column names clean text
df["clean_text"] = df["review_text"].apply(preprocess_text)


# splitting data to train and test datasets
train_df = df[df["fold"] < 5].reset_index(drop=True) # folds 1-4 for training (640 reviews)
test_df  = df[df["fold"] == 5].reset_index(drop=True) # fold 5 for testing (160 reviews)

# Extract labels (same for both unigram and bigram features)
y_train = train_df["label"]
y_test = test_df["label"]


# Feature extraction

#Unigrams only
tfidf_uni = TfidfVectorizer(
        max_df=0.95,  # removes terms appearing in >95% of documents
        min_df=2, # Remove terms appearing in <2 documents
        ngram_range=(1,1), #unigrams (1 word)
        lowercase=False,  # Already lowercased, so needed to do it again
        strip_accents=None
)


X_train_uni = tfidf_uni.fit_transform(train_df["clean_text"])
X_test_uni= tfidf_uni.transform(test_df["clean_text"]) 


#Unigrams and Bigrams
tfidf_bi = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),# unigrams and bigrams
        lowercase=False,
        strip_accents=None
)

X_train_bi = tfidf_bi.fit_transform(train_df["clean_text"])
X_test_bi = tfidf_bi.transform(test_df["clean_text"])

# Save dataframe with clean text
df.to_csv("processed_reviews.csv", index=False) #saves whole dataframe (with original and clean text) to csv file

# Save train/test splits
train_df.to_csv("train_reviews.csv", index=False)
test_df.to_csv("test_reviews.csv", index=False)

#save vectorizers 
with open("tfidf_uni.pkl", "wb") as f:
    pickle.dump(tfidf_uni, f)
with open("tfidf_bi.pkl", "wb") as f:
    pickle.dump(tfidf_bi, f)