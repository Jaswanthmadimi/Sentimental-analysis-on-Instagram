import pandas as pd
from preprocessing import preprocess_text, is_english
from sentiment_analysis import get_textblob_sentiment
from visualization import plot_sentiment_distribution, plot_wordcloud, plot_time_sentiment_trend

def main():
    data_file = 'sample_instagram_data.csv'
    
    # Step 1: Load data from dataset
    print("Loading Instagram dataset...")
    df = pd.read_csv(data_file)
    
    # Step 2: Preprocess captions and comments
    print("Preprocessing text data...")
    df['caption_clean'] = df['caption'].fillna('').apply(preprocess_text)
    
    # Filter English captions only
    df['is_english'] = df['caption_clean'].apply(is_english)
    df = df[df['is_english']]
    
    # Step 3: Sentiment analysis on captions
    print("Analyzing sentiment...")
    sentiments = df['caption_clean'].apply(get_textblob_sentiment)
    df['polarity'] = sentiments.apply(lambda x: x[0])
    df['sentiment'] = sentiments.apply(lambda x: x[1])
    
    # Step 4: Visualization
    print("Visualizing results...")
    plot_sentiment_distribution(df, sentiment_col='sentiment')
    
    # Wordclouds for positive and negative captions
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['caption_clean'])
    negative_text = ' '.join(df[df['sentiment'] == 'negative']['caption_clean'])
    
    if positive_text:
        plot_wordcloud(positive_text, 'Positive Captions Word Cloud')
    if negative_text:
        plot_wordcloud(negative_text, 'Negative Captions Word Cloud')
    
    # Time-based sentiment trend
    plot_time_sentiment_trend(df, date_col='post_date', sentiment_col='sentiment')

if __name__ == "__main__":
    main()
