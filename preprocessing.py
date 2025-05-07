import re
import spacy
from langdetect import detect
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """
    Clean text by removing emojis, special characters, URLs, hashtags, and extra spaces.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove emojis and non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove special characters except spaces
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Preprocess text: clean, tokenize, lemmatize, remove stopwords.
    Returns cleaned and lemmatized text.
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

def is_english(text):
    """
    Detect if the text is English.
    """
    try:
        return detect(text) == 'en'
    except:
        return False

if __name__ == "__main__":
    sample_text = "This is a sample text! 😊 #hashtag http://example.com"
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
    print("Preprocessed:", preprocess_text(sample_text))
    print("Is English:", is_english(sample_text))
