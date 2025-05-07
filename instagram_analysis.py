import pandas as pd
from preprocessing import preprocess_text, is_english
from sentiment_analysis import get_textblob_sentiment, get_vader_sentiment
import visualization

def analyze_comments():
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
    print("Plotting TextBlob sentiment distribution for comments...")
    visualization.plot_sentiment_distribution(df, sentiment_col='textblob_sentiment')

    # Visualize sentiment distribution for VADER
    print("Plotting VADER sentiment distribution for comments...")
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
        print("Visualizing sentiment vs likes for comments...")
        df.boxplot(column='likes', by='textblob_sentiment', grid=False)
        import matplotlib.pyplot as plt
        plt.title('Likes Distribution by TextBlob Sentiment (Comments)')
        plt.suptitle('')
        plt.xlabel('Sentiment')
        plt.ylabel('Likes')
        plt.show()

    # Optional: Visualize sentiment vs hashtags count
    if 'hashtags_used_count' in df.columns:
        print("Visualizing sentiment vs hashtags count for comments...")
        df.boxplot(column='hashtags_used_count', by='textblob_sentiment', grid=False)
        plt.title('Hashtags Count Distribution by TextBlob Sentiment (Comments)')
        plt.suptitle('')
        plt.xlabel('Sentiment')
        plt.ylabel('Hashtags Count')
        plt.show()

def main():
    # Analyze captions
    print("Analyzing Instagram captions...")
    # Load Instagram review data
    df = pd.read_csv('sample_instagram_data.csv', encoding='utf-8')

    # Filter English reviews only
    df['is_english'] = df['caption'].apply(is_english)
    df = df[df['is_english']]

    # Preprocess the content
    df['preprocessed_content'] = df['caption'].apply(preprocess_text)

    # Perform sentiment analysis using TextBlob
    df[['textblob_polarity', 'textblob_sentiment']] = df['preprocessed_content'].apply(
        lambda x: pd.Series(get_textblob_sentiment(x))
    )

    # Perform sentiment analysis using VADER
    df[['vader_compound', 'vader_sentiment']] = df['preprocessed_content'].apply(
        lambda x: pd.Series(get_vader_sentiment(x))
    )

    # Display sentiment distribution plots for TextBlob
    print("Plotting TextBlob sentiment distribution for captions...")
    visualization.plot_sentiment_distribution(df, sentiment_col='textblob_sentiment')

    # Display sentiment distribution plots for VADER
    print("Plotting VADER sentiment distribution for captions...")
    visualization.plot_sentiment_distribution(df, sentiment_col='vader_sentiment')

    # Generate word cloud for positive reviews (TextBlob)
    positive_text = ' '.join(df[df['textblob_sentiment'] == 'positive']['preprocessed_content'])
    if positive_text:
        print("Generating word cloud for positive reviews (TextBlob)...")
        visualization.plot_wordcloud(positive_text, 'Word Cloud - Positive Reviews (TextBlob)')

    # Generate word cloud for negative reviews (TextBlob)
    negative_text = ' '.join(df[df['textblob_sentiment'] == 'negative']['preprocessed_content'])
    if negative_text:
        print("Generating word cloud for negative reviews (TextBlob)...")
        visualization.plot_wordcloud(negative_text, 'Word Cloud - Negative Reviews (TextBlob)')

    # Analyze comments
    print("Analyzing Instagram comments...")
    analyze_comments()

if __name__ == "__main__":
    main()
