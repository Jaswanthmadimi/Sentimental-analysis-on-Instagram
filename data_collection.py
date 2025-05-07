import instaloader
import pandas as pd
import time
from datetime import datetime

def collect_instagram_data(profile_name, max_posts=50, output_csv='instagram_data.csv', username=None, password=None):
    """
    Collect posts, captions, comments, hashtags, and metadata from an Instagram profile using Instaloader.
    Supports optional login to avoid rate limits.
    Saves the collected data to a CSV file.
    
    Args:
        profile_name (str): Instagram profile username to scrape.
        max_posts (int): Maximum number of posts to scrape.
        output_csv (str): Path to save the collected data CSV.
        username (str): Instagram login username (optional).
        password (str): Instagram login password (optional).
    """
    L = instaloader.Instaloader(download_comments=True, save_metadata=False, download_videos=False, download_video_thumbnails=False)
    
    if username and password:
        print(f"Logging in as {username}")
        L.login(username, password)
    else:
        print("No login credentials provided, proceeding anonymously (may hit rate limits).")
    
    print(f"Loading profile: {profile_name}")
    profile = instaloader.Profile.from_username(L.context, profile_name)
    
    data = []
    post_count = 0
    
    for post in profile.get_posts():
        if post_count >= max_posts:
            break
        
        retry_count = 0
        max_retries = 5
        delay = 10
        
        while retry_count < max_retries:
            try:
                post_date = post.date_utc.strftime('%Y-%m-%d %H:%M:%S')
                caption = post.caption if post.caption else ""
                hashtags = post.caption_hashtags if post.caption_hashtags else []
                likes = post.likes
                comments = []
                
                # Collect comments text
                for comment in post.get_comments():
                    comments.append(comment.text)
                
                data.append({
                    'post_id': post.mediaid,
                    'post_date': post_date,
                    'caption': caption,
                    'hashtags': ','.join(hashtags),
                    'likes': likes,
                    'comments': ' ||| '.join(comments)  # separate comments by |||
                })
                
                post_count += 1
                print(f"Collected post {post_count}/{max_posts}")
                break
            except Exception as e:
                retry_count += 1
                print(f"Error collecting post: {e}. Retry {retry_count}/{max_retries} after {delay} seconds...")
                time.sleep(delay)
                delay *= 2
        else:
            print(f"Failed to collect post after {max_retries} retries, skipping.")
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    # Example usage: collect data from Instagram profile 'instagram' with max 20 posts
    collect_instagram_data('instagram', max_posts=20)
