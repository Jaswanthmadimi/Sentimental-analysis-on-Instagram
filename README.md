# Sentimental Analysis on Instagram Using NLP

## Project Description
This project performs sentiment analysis on Instagram data using Natural Language Processing (NLP) techniques. It collects Instagram posts and comments, preprocesses the text, analyzes sentiment using two popular methods (TextBlob and VADER), and visualizes the results through various plots and word clouds.

## Features
- Collect Instagram posts, captions, comments, hashtags, likes, and metadata using Instaloader.
- Preprocess text data to clean and filter English content.
- Perform sentiment analysis using TextBlob and VADER.
- Visualize sentiment distribution, word clouds for positive and negative sentiments, and time-based sentiment trends.
- Analyze both Instagram captions and comments.
- Optional visualizations of sentiment versus likes and hashtags count.

## Installation
Install the required Python packages using:

```
pip install -r requirements.txt
```

## Usage

### Data Collection
Use `data_collection.py` to scrape Instagram data from a profile.

```bash
python data_collection.py
```

You can customize the profile name, number of posts, and login credentials in the script or by modifying the function call.

### Sentiment Analysis and Visualization
- Use `main.py` to analyze sentiment on Instagram captions and visualize the results.
- Use `instagram_analysis.py` to analyze sentiment on both captions and comments with additional visualizations.

Run the scripts with:

```bash
python main.py
```

or

```bash
python instagram_analysis.py
```

## File Descriptions
- `data_collection.py`: Collects Instagram data using Instaloader.
- `main.py`: Main script for sentiment analysis and visualization on captions.
- `instagram_analysis.py`: Extended analysis including comments and multiple sentiment methods.
- `sentiment_analysis.py`: Implements sentiment analysis functions using TextBlob and VADER.
- `preprocessing.py`: Contains text preprocessing and language detection functions.
- `visualization.py`: Contains functions to plot sentiment distributions, word clouds, and trends.
- CSV files (e.g., `comments.csv`, `sample_instagram_data.csv`): Datasets used for analysis.

## Requirements
- instaloader
- pandas
- numpy
- spacy
- textblob
- matplotlib
- seaborn
- wordcloud
- langdetect

## Example
Collect data from an Instagram profile and analyze sentiment:

```python
# Collect data (example)
from data_collection import collect_instagram_data
collect_instagram_data('instagram', max_posts=20)

# Analyze and visualize
python instagram_analysis.py
```

## License
This project is open source and free to use.
