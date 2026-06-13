import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# Load stopwords
stop_words = set(stopwords.words("english"))

# Keep important negation words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("never")


def preprocess_text(text):
    """
    Clean and preprocess text.

    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove special characters
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep letters and spaces only
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)