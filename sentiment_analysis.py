from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_textblob_sentiment(text):
    """
    Get sentiment polarity using TextBlob.
    Returns polarity score (-1 to 1) and sentiment label.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment = 'positive'
    elif polarity < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return polarity, sentiment

def get_vader_sentiment(text):
    """
    Get sentiment polarity using VADER.
    Returns compound score (-1 to 1) and sentiment label.
    """
    vs = analyzer.polarity_scores(text)
    compound = vs['compound']
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return compound, sentiment

if __name__ == "__main__":
    sample_text = "I love this product! It's amazing."
    print("TextBlob:", get_textblob_sentiment(sample_text))
    print("VADER:", get_vader_sentiment(sample_text))
