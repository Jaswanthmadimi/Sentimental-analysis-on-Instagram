import pandas as pd
from preprocessing import preprocess_text, is_english
from sentiment_analysis import get_textblob_sentiment, get_vader_sentiment
import visualization

def main():
    # Load Instagram comments data
    df = pd.read_csv('comments.csv', encoding='utf-8')

    # Filter English comments only
    df['is_english'] = df['comment'].apply(is_english)
    df = df[df['is_english']]

    # Preprocess the comments
    df['preprocessed_comment'] = df['comment'].apply(preprocess_text)

    # Perform sentiment analysis using TextBlob
    df[['textblob_polarity', 'textblob_sentiment']] = df['preprocessed_comment'].apply(
        lambda x: pd.Series(get_textblob_sentiment(x))
    )

    # Perform sentiment analysis using VADER
    df[['vader_compound', 'vader_sentiment']] = df['preprocessed_comment'].apply(
        lambda x: pd.Series(get_vader_sentiment(x))
    )

    # Visualize sentiment distribution for TextBlob
    print("Plotting TextBlob sentiment distribution...")
    visualization.plot_sentiment_distribution(df, sentiment_col='textblob_sentiment')

    # Visualize sentiment distribution for VADER
    print("Plotting VADER sentiment distribution...")
    visualization.plot_sentiment_distribution(df, sentiment_col='vader_sentiment')

    # Generate word cloud for positive comments (TextBlob)
    positive_text = ' '.join(df[df['textblob_sentiment'] == 'positive']['preprocessed_comment'])
    if positive_text:
        print("Generating word cloud for positive comments (TextBlob)...")
        visualization.plot_wordcloud(positive_text, 'Word Cloud - Positive Comments (TextBlob)')

    # Generate word cloud for negative comments (TextBlob)
    negative_text = ' '.join(df[df['textblob_sentiment'] == 'negative']['preprocessed_comment'])
    if negative_text:
        print("Generating word cloud for negative comments (TextBlob)...")
        visualization.plot_wordcloud(negative_text, 'Word Cloud - Negative Comments (TextBlob)')

    # Optional: Visualize sentiment vs likes
    if 'likes' in df.columns:
        print("Visualizing sentiment vs likes...")
        df.boxplot(column='likes', by='textblob_sentiment', grid=False)
        import matplotlib.pyplot as plt
        plt.title('Likes Distribution by TextBlob Sentiment')
        plt.suptitle('')
        plt.xlabel('Sentiment')
        plt.ylabel('Likes')
        plt.show()

    # Optional: Visualize sentiment vs hashtags count
    if 'hashtags_used_count' in df.columns:
        print("Visualizing sentiment vs hashtags count...")
        df.boxplot(column='hashtags_used_count', by='textblob_sentiment', grid=False)
        plt.title('Hashtags Count Distribution by TextBlob Sentiment')
        plt.suptitle('')
        plt.xlabel('Sentiment')
        plt.ylabel('Hashtags Count')
        plt.show()

if __name__ == "__main__":
    main()
