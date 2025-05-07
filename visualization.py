import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_sentiment_distribution(df, sentiment_col='sentiment'):
    """
    Plot bar chart of sentiment distribution.
    """
    plt.figure(figsize=(6,4))
    sns.countplot(x=sentiment_col, data=df, order=['positive', 'neutral', 'negative'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

def plot_wordcloud(text, title):
    """
    Generate and display a word cloud from text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def plot_time_sentiment_trend(df, date_col='post_date', sentiment_col='sentiment'):
    """
    Plot time-based sentiment trends.
    Assumes df[date_col] is datetime type.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df_grouped = df.groupby([pd.Grouper(key=date_col, freq='D'), sentiment_col]).size().unstack(fill_value=0)
    df_grouped.plot(kind='line', figsize=(10,5))
    plt.title('Sentiment Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.show()
