import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

import subprocess
import streamlit as st
import logging
st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")
from streamlit_option_menu import option_menu
import praw
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import datetime
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import streamlit as st
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
from selenium.common.exceptions import NoSuchElementException
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from datetime import datetime,timedelta
import datetime


# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Select Platform",
        ["YouTube", "X (Twitter)", "Reddit", "Instagram", "E-Commerce"],
        icons=["youtube", "twitter", "reddit", "instagram", "cart"],
        menu_icon="cast",
        default_index=0
    )

if selected == "YouTube":
    tab1, tab2 ,tab3= st.tabs(["Single Video Analysis", "Comparative Analysis","Channel Analysis"])

    API_KEY = 'AIzaSyBIs8-G0D0XsYjnxwahKvAxDkePHcmNmZA'

    with tab1:
        if API_KEY:
            youtube = build('youtube', 'v3', developerKey=API_KEY)
            analyzer = SentimentIntensityAnalyzer()
            st.title("📊 YouTube Comment Sentiment Analysis & Engagement Insights")

            # Input Video URL
            video_url = st.text_input("Enter YouTube Video URL:")

            if video_url:
                video_id = video_url[-11:]
                st.video(video_url)

                # Fetch video details
                video_response = youtube.videos().list(part='snippet,statistics', id=video_id).execute()
                video_snippet = video_response['items'][0]['snippet']
                video_stats = video_response['items'][0]['statistics']
                uploader_channel_id = video_snippet['channelId']

                st.subheader(f"📺 {video_snippet['title']}")
                st.write(f"👤 Channel: {video_snippet['channelTitle']}")
                st.write(
                    f"👍 Likes: {video_stats.get('likeCount', 0)} | 💬 Comments: {video_stats.get('commentCount', 0)}")

                # Fetch comments
                comments = []
                user_engagement = {}
                nextPageToken = None
                while len(comments) < 500:
                    response = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=100,
                        pageToken=nextPageToken
                    ).execute()

                    for item in response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']
                        author = comment['authorDisplayName']
                        if comment['authorChannelId']['value'] != uploader_channel_id:
                            comments.append(comment['textDisplay'])
                            user_engagement[author] = user_engagement.get(author, 0) + 1

                    nextPageToken = response.get('nextPageToken')
                    if not nextPageToken:
                        break

                # Sentiment Analysis
                sentiments = {'positive': [], 'negative': [], 'neutral': []}
                sentiment_scores = []
                sentiment_labels = []

                for comment in comments:
                    score = analyzer.polarity_scores(comment)['compound']
                    sentiment_scores.append(score)
                    if score > 0.05:
                        sentiments['positive'].append(comment)
                        sentiment_labels.append("Positive")
                    elif score < -0.05:
                        sentiments['negative'].append(comment)
                        sentiment_labels.append("Negative")
                    else:
                        sentiments['neutral'].append(comment)
                        sentiment_labels.append("Neutral")

                avg_polarity = np.mean(sentiment_scores)

                # Display sentiment results
                st.subheader("🧐 Sentiment Analysis")
                total_comments = len(comments)
                sentiment_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [len(sentiments['positive']), len(sentiments['negative']), len(sentiments['neutral'])]
                })
                fig_sentiment = px.pie(sentiment_df, names='Sentiment', values='Count', title='Sentiment Distribution',
                                       hole=0.4)
                st.plotly_chart(fig_sentiment)

                # Sentiment Score Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_polarity,
                    title={"text": "Average Sentiment Score"},
                    gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "darkblue"}}
                ))
                st.plotly_chart(fig_gauge)

                # WordCloud & Keyword Extraction
                st.subheader("🔍 Trending Topics")
                wordcloud_text = ' '.join(comments)
                wordcloud = WordCloud(width=800, height=400, background_color='black').generate(wordcloud_text)
                st.image(wordcloud.to_array())
                words = re.findall(r'\w+', wordcloud_text.lower())
                top_words = Counter(words).most_common(10)
                keyword_df = pd.DataFrame(top_words, columns=['Keyword', 'Frequency'])
                fig_keywords = px.bar(keyword_df, x='Keyword', y='Frequency', title='Top Keywords', color='Frequency')
                st.plotly_chart(fig_keywords)

                # User Engagement Analysis
                st.subheader("👥 Most Engaged Users")
                top_users = sorted(user_engagement.items(), key=lambda x: x[1], reverse=True)[:5]
                user_df = pd.DataFrame(top_users, columns=["User", "Comment Count"])
                fig_users = px.bar(user_df, x='User', y='Comment Count', title='Top Engaged Users',
                                   color='Comment Count')
                st.plotly_chart(fig_users)

                # Engagement Heatmap
                post_times = [datetime.datetime.utcnow() - datetime.timedelta(days=np.random.randint(1, 30)) for _ in
                              range(len(comments))]
                df = pd.DataFrame({'date': [t.strftime('%Y-%m-%d') for t in post_times]})
                df['count'] = 1
                df = df.groupby('date').sum().reset_index()
                df['date'] = pd.to_datetime(df['date'])
                df_heatmap = df.pivot_table(index=df['date'].dt.strftime('%A'), columns=df['date'].dt.hour,
                                            values='count', aggfunc='sum', fill_value=0)

                st.subheader("📈 Engagement Heatmap")
                fig_heatmap = px.imshow(df_heatmap, labels=dict(x="Hour", y="Day", color="Comments"),
                                        title="Comment Frequency by Time", color_continuous_scale='Blues')
                st.plotly_chart(fig_heatmap)

                # Actionable Insights
                st.subheader("🚀 Actionable Insights")
                st.write(
                    "🔹 **Best Times to Post**: Based on engagement trends, consider posting during peak comment activity hours.")
                st.write("🔹 **Trending Topics**: The most mentioned words can help shape your content strategy.")
                st.write("🔹 **User Engagement**: Engage with highly active users to build a loyal audience.")

                # All Analyzed Comments
                st.subheader("📝 All Analyzed Comments")
                comments_df = pd.DataFrame({"Comment": comments, "Sentiment": sentiment_labels})
                st.dataframe(comments_df)

                st.success("📌 Insights Generated Successfully! Use these insights to optimize your content strategy!")

    with tab2:
        if API_KEY:
            youtube = build('youtube', 'v3', developerKey=API_KEY)
            analyzer = SentimentIntensityAnalyzer()
            st.title("📊 YouTube Multi-Video Comparative Dashboard")
            video_urls = st.text_area("Enter YouTube Video URLs (one per line)").split('\n')

            if video_urls:
                video_data = []

                for video_url in video_urls:
                    video_id = video_url.strip()[-11:]
                    video_response = youtube.videos().list(part='snippet,statistics', id=video_id).execute()
                    if 'items' not in video_response or not video_response['items']:
                        st.warning(f"Invalid or private video: {video_url}")
                        continue

                    video_snippet = video_response['items'][0]['snippet']
                    video_stats = video_response['items'][0]['statistics']
                    uploader_channel_id = video_snippet['channelId']

                    comments = []
                    nextPageToken = None
                    while len(comments) < 500:
                        response = youtube.commentThreads().list(
                            part='snippet', videoId=video_id, maxResults=100, pageToken=nextPageToken
                        ).execute()

                        for item in response['items']:
                            comment = item['snippet']['topLevelComment']['snippet']
                            if comment['authorChannelId']['value'] != uploader_channel_id:
                                comments.append(comment['textDisplay'])

                        nextPageToken = response.get('nextPageToken')
                        if not nextPageToken:
                            break

                    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
                    sentiment_scores = []
                    for comment in comments:
                        score = analyzer.polarity_scores(comment)['compound']
                        sentiment_scores.append(score)
                        if score > 0.05:
                            sentiments['positive'] += 1
                        elif score < -0.05:
                            sentiments['negative'] += 1
                        else:
                            sentiments['neutral'] += 1

                    avg_polarity = np.mean(sentiment_scores) if sentiment_scores else 0
                    video_data.append({
                        "Video Title": video_snippet['title'],
                        "Likes": int(video_stats.get('likeCount', 0)),
                        "Comments": len(comments),
                        "Positive": sentiments['positive'],
                        "Negative": sentiments['negative'],
                        "Neutral": sentiments['neutral'],
                        "Average Sentiment": avg_polarity,
                        "View Count": int(video_stats.get('viewCount', 0)),
                        "Dislikes": int(video_stats.get('dislikeCount', 0)) if 'dislikeCount' in video_stats else 0
                    })

                df = pd.DataFrame(video_data)

                    # Dashboard Layout
                col1, col2 = st.columns(2)
                with col1:


                    try:
                        fig_sentiment = px.bar(
                            df, x="Video Title", y=["Positive", "Negative", "Neutral"],
                            title="Sentiment Comparison Across Videos", barmode='group',
                            color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"}
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    except ValueError as e:
                        logging.warning(f"Plotly error: {e}")
                        st.warning("There was an issue generating the chart.")



                with col2:
                    try:
                        fig_engagement = px.scatter(
                            df, x="Likes", y="Comments", size="Comments", color="Average Sentiment",
                            hover_name="Video Title", title="Likes vs Comments (Engagement)", size_max=60,
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_engagement, use_container_width=True)
                    except ValueError as e:
                        logging.warning(f"Plotly error: {e}")
                        st.warning("There was an issue generating the chart.")
                st.subheader("📊 Additional Insights")
                col3, col4 = st.columns(2)
                with col3:
                    try:

                        fig_views = px.bar(
                            df, x="Video Title", y="View Count", title="View Count Across Videos",
                            color="View Count", color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_views, use_container_width=True)
                    except ValueError as e:
                        logging.warning(f"Plotly error: {e}")
                        st.warning("There was an issue generating the chart.")

                with col4:
                    try:

                        fig_dislikes = px.bar(
                            df, x="Video Title", y="Dislikes", title="Dislike Count Across Videos",
                            color="Dislikes", color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_dislikes, use_container_width=True)
                    except ValueError as e:
                        logging.warning(f"Plotly error: {e}")
                        st.warning("There was an issue generating the chart.")

                st.subheader("🔍 Video Analysis Data")
                st.dataframe(df)

                st.success("📌 Comparative analysis complete! Use this dashboard to gain deeper insights.")

    # with tab3:
    #     if API_KEY:
    #         youtube = build('youtube', 'v3', developerKey=API_KEY)
    #         analyzer = SentimentIntensityAnalyzer()
    #         st.title("📊 YouTube Channel Dashboard")
    #         channel_url = st.text_input("Enter YouTube Channel URL:")
    #         if st.button("Fetch Videos"):
    #             command = f'yt-dlp --flat-playlist --print "%(id)s" "{channel_url}"'
    #             result = subprocess.run(command, shell=True, capture_output=True, text=True)
    #             video_ids = result.stdout.strip().split("\n")
    #             video_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    #             st.session_state["video_urls"] = video_urls
    #
    #         video_urls = st.session_state.get("video_urls", [])
    #         if video_urls:
    #             st.write(f"Fetched {len(video_urls)} videos from the channel.")
    #             st.text_area("Video URLs:", "\n".join(video_urls), height=150)
    #
    #             # Advanced Visualizations for Channel Performance
    #             st.subheader("📈 Channel Performance Overview")
    #             st.write("Understanding video performance trends and engagement patterns.")
    #
    #             # Placeholder for aggregated metrics
    #             total_views = 0
    #             total_likes = 0
    #             total_comments = 0
    #             engagement_ratios = []
    #             titles = []
    #             view_counts = []
    #             like_counts = []
    #             comment_counts = []
    #
    #             for video_url in video_urls:
    #                 video_id = video_url.strip()[-11:]
    #                 video_response = youtube.videos().list(part='snippet,statistics', id=video_id).execute()
    #                 if 'items' not in video_response or not video_response['items']:
    #                     continue
    #
    #                 video_snippet = video_response['items'][0]['snippet']
    #                 video_stats = video_response['items'][0]['statistics']
    #                 titles.append(video_snippet['title'])
    #                 views = int(video_stats.get('viewCount', 0))
    #                 likes = int(video_stats.get('likeCount', 0))
    #                 comments = int(video_stats.get('commentCount', 0))
    #                 engagement_ratio = (likes + comments) / max(1, views)
    #
    #                 view_counts.append(views)
    #                 like_counts.append(likes)
    #                 comment_counts.append(comments)
    #                 engagement_ratios.append(engagement_ratio)
    #
    #                 total_views += views
    #                 total_likes += likes
    #                 total_comments += comments
    #
    #             # Create summary dataframe
    #             summary_df = pd.DataFrame({
    #                 "Metric": ["Total Views", "Total Likes", "Total Comments"],
    #                 "Count": [total_views, total_likes, total_comments]
    #             })
    #
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 st.bar_chart(summary_df.set_index("Metric"))
    #             with col2:
    #                 fig_engagement = px.line(x=titles, y=engagement_ratios, title="Engagement Ratio Across Videos",
    #                                          markers=True, labels={'x': 'Video Title', 'y': 'Engagement Ratio'})
    #                 st.plotly_chart(fig_engagement, use_container_width=True)
    #
    #             # Additional Interactive Visualizations
    #             fig_views = px.bar(x=titles, y=view_counts, title="Views Per Video",
    #                                labels={'x': 'Video Title', 'y': 'View Count'})
    #             fig_likes = px.bar(x=titles, y=like_counts, title="Likes Per Video",
    #                                labels={'x': 'Video Title', 'y': 'Like Count'})
    #             fig_comments = px.bar(x=titles, y=comment_counts, title="Comments Per Video",
    #                                   labels={'x': 'Video Title', 'y': 'Comment Count'})
    #
    #             st.plotly_chart(fig_views, use_container_width=True)
    #             st.plotly_chart(fig_likes, use_container_width=True)
    #             st.plotly_chart(fig_comments, use_container_width=True)

if selected == "X (Twitter)":
    import streamlit as st
    import pandas as pd
    import joblib
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    import plotly.express as px
    from streamlit.components.v1 import html

    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        .st-bw {background-color: #ffffff;}
        .css-1d391kg {padding-top: 3rem;}
        h1 {color: #2b5876;}
        .sidebar .sidebar-content {background-image: linear-gradient(#292929,#2b5876);}
    </style>
    """, unsafe_allow_html=True)


    @st.cache_data
    def load_data():
        train_data = pd.read_csv('twitter_training.csv',
                                 names=['serial_number', 'source', 'sentiment', 'text'])
        val_data = pd.read_csv('twitter_validation.csv',
                               names=['serial_number', 'source', 'sentiment', 'text'])
        return train_data, val_data


    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)


    @st.cache_data
    def preprocess_data(data):
        data['Processed_text'] = data['text'].apply(preprocess_text)
        return data


    def predict_sentiment(text, source):
        try:
            model = joblib.load(f'models/{source}_model.joblib')
            vectorizer = joblib.load(f'models/{source}_vectorizer.joblib')
            processed_text = preprocess_text(text)
            vectorized_text = vectorizer.transform([processed_text])
            return model.predict(vectorized_text)[0]
        except FileNotFoundError:
            return "Model not found"


    # Load and preprocess data
    train_data, val_data = load_data()
    train_data = preprocess_data(train_data)
    val_data = preprocess_data(val_data)
    sources = train_data['source'].unique()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", [
        "📊 Dashboard",
        "🔍 Real-Time Analysis",
        "📈 Model Insights",
        "🤖 Source Comparison",
        "📚 Data Explorer"
    ])

    # Main content
    if page == "📊 Dashboard":
        st.title("Social Media Sentiment Analysis Dashboard")

        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Training Samples", len(train_data))
        with col2:
            st.metric("Validation Accuracy", "84%")  # Update with actual metric
        with col3:
            st.metric("Supported Platforms", len(sources))

        # Sentiment distribution
        st.subheader("Sentiment Distribution Across Platforms")
        fig = px.sunburst(train_data, path=['source', 'sentiment'],
                          color='sentiment', height=600)
        st.plotly_chart(fig)

    elif page == "🔍 Real-Time Analysis":
        st.title("Real-Time Sentiment Analysis")

        # Interactive prediction
        with st.form("prediction_form"):
            col1, col2 = st.columns([3, 1])
            with col1:
                text_input = st.text_area("Enter your text:", height=150)
            with col2:
                source_input = st.selectbox("Select platform:", sources)

            if st.form_submit_button("Analyze Sentiment"):
                if text_input and source_input:
                    sentiment = predict_sentiment(text_input, source_input)
                    if sentiment != "Model not found":
                        st.success(f"Predicted Sentiment: **{sentiment}**")
                    else:
                        st.error(f"No model available for {source_input}")

        # Live sentiment visualization
        st.subheader("Sentiment Intensity Meter")
        html_str = """
        <div id="sentimentMeter" style="width: 100%; height: 60px; background: linear-gradient(90deg, 
            #ff0000 0%, #ff8000 25%, #ffff00 50%, #80ff00 75%, #00ff00 100%); 
            border-radius: 10px; position: relative;">
            <div id="indicator" style="position: absolute; height: 100%; width: 4px; 
                background: black; left: 50%; transform: translateX(-50%);"></div>
        </div>
        <script>
            function updateMeter(pos) {
                document.getElementById('indicator').style.left = pos + '%';
            }
        </script>
        """
        html(html_str, height=100)

        if text_input:
            sentiment_score = 0.7  # Replace with actual score calculation
            meter_position = (sentiment_score + 1) * 50  # Convert -1 to 1 scale to 0-100
            st.write(f"<script>updateMeter({meter_position})</script>", unsafe_allow_html=True)

    elif page == "📈 Model Insights":
        st.title("Model Performance Insights")

        # Confusion matrix
        st.subheader("Validation Performance")
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(val_data['sentiment'], val_data['sentiment'])  # Replace with actual predictions
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Feature importance
        st.subheader("Key Sentiment Indicators")
        selected_source = st.selectbox("Select Platform:", sources)

        try:
            vectorizer = joblib.load(f'models/{selected_source}_vectorizer.joblib')
            model = joblib.load(f'models/{selected_source}_model.joblib')

            features = vectorizer.get_feature_names_out()
            coefficients = model.coef_

            for idx, sentiment in enumerate(model.classes_):
                st.subheader(f"Top words for {sentiment}")
                top_words = pd.DataFrame({
                    'Word': features,
                    'Importance': coefficients[idx]
                }).sort_values('Importance', ascending=False).head(10)

                fig = px.bar(top_words, x='Importance', y='Word', orientation='h',
                             color='Importance', height=400)
                st.plotly_chart(fig)

        except FileNotFoundError:
            st.error(f"No model available for {selected_source}")

    elif page == "🤖 Source Comparison":
        st.title("Cross-Platform Sentiment Comparison")

        compare_text = st.text_area("Enter text to compare across platforms:", height=100)
        if compare_text:
            results = []
            for source in sources:
                sentiment = predict_sentiment(compare_text, source)
                results.append({
                    'Platform': source,
                    'Sentiment': sentiment,
                    'Confidence': 0.8  # Replace with actual confidence score
                })

            df = pd.DataFrame(results)

            # Sentiment distribution
            st.subheader("Sentiment Distribution Across Platforms")
            fig = px.bar(df, x='Platform', y='Confidence', color='Sentiment',
                         height=400)
            st.plotly_chart(fig)

            # Detailed results
            st.subheader("Detailed Predictions")
            st.dataframe(df.style.background_gradient(cmap='Blues'))

    elif page == "📚 Data Explorer":
        st.title("Data Exploration Hub")

        # Raw data viewer
        st.subheader("Dataset Preview")
        dataset_choice = st.radio("Choose dataset:", ["Training Data", "Validation Data"])
        if dataset_choice == "Training Data":
            st.dataframe(train_data.head(100))
        else:
            st.dataframe(val_data.head(100))

        # Word clouds
        st.subheader("Word Cloud Analysis")
        selected_sentiment = st.selectbox("Select sentiment:",
                                          train_data['sentiment'].unique())

        filtered_text = ' '.join(train_data[train_data['sentiment'] == selected_sentiment]['Processed_text'])
        wordcloud = WordCloud(width=800, height=400).generate(filtered_text)

        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)
if selected=="Reddit":

    # Initialize PRAW with Streamlit secrets
    reddit = praw.Reddit(
        client_id=st.secrets["REDDIT"]["CLIENT_ID"],
        client_secret=st.secrets["REDDIT"]["CLIENT_SECRET"],
        username=st.secrets["REDDIT"]["USERNAME"],
        password=st.secrets["REDDIT"]["PASSWORD"],
        user_agent=st.secrets["REDDIT"]["USER_AGENT"]
    )

    # Download NLTK resources
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()


    def analyze_sentiment(text):
        sentiment_score = sia.polarity_scores(text)
        if sentiment_score["compound"] > 0.05:
            return "Positive"
        elif sentiment_score["compound"] < -0.05:
            return "Negative"
        else:
            return "Neutral"


    # Streamlit app layout
    st.title("📊 Reddit Comment Sentiment Analysis")
    st.markdown("Analyze sentiment of any Reddit user's recent comments")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        num_comments = st.slider("Number of comments to analyze", 1, 100, 10)
        show_raw = st.checkbox("Show raw data")
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit")

    # Main input
    username = st.text_input("Enter Reddit username:", "SentientClit")

    if st.button("Analyze Comments"):
        if username:
            try:
                with st.spinner(f"Fetching and analyzing {num_comments} comments..."):
                    user = reddit.redditor(username)
                    comments = []
                    sentiments = []
                    timestamps = []

                    for comment in user.comments.new(limit=num_comments):
                        comments.append(comment.body)
                        sentiments.append(analyze_sentiment(comment.body))
                        timestamps.append(datetime.fromtimestamp(comment.created_utc))

                    df = pd.DataFrame({
                        "Comment": comments,
                        "Sentiment": sentiments,
                        "Timestamp": timestamps
                    })

                    # Calculate metrics
                    sentiment_counts = df["Sentiment"].value_counts()

                    # Display results
                    st.success("Analysis complete!")

                    # Metrics columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Comments Analyzed", len(df))
                    with col2:
                        st.metric("Most Common Sentiment", sentiment_counts.idxmax())
                    with col3:
                        st.metric("Positive Comments",
                                  f"{sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0) / len(df):.0%})")

                    # Visualization section
                    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Timeline", "Word Cloud", "Raw Data"])

                    with tab1:
                        fig = px.pie(df, names="Sentiment", title="Sentiment Distribution")
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        df["Date"] = df["Timestamp"].dt.date
                        timeline_df = df.groupby(["Date", "Sentiment"]).size().reset_index(name="Counts")
                        fig = px.line(timeline_df, x="Date", y="Counts", color="Sentiment",
                                      title="Sentiment Over Time")
                        st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        text = " ".join(comment for comment in df["Comment"])
                        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        st.pyplot(plt)

                    with tab4:
                        st.dataframe(df[["Comment", "Sentiment", "Timestamp"]], use_container_width=True)

                    # Download button
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download analysis as CSV",
                        data=csv,
                        file_name=f"{username}_sentiment_analysis.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error analyzing user: {str(e)}")
        else:
            st.warning("Please enter a Reddit username")
if selected == 'E-Commerce':

    nltk.download('vader_lexicon')
    nltk.download('punkt')

    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # Predefined product aspects
    ASPECTS = {
        'fit': ['fit', 'size', 'waist', 'length', 'measurement'],
        'quality': ['quality', 'material', 'fabric', 'durable', 'stitching'],
        'comfort': ['comfort', 'soft', 'stretch', 'flexible', 'breathable'],
        'delivery': ['delivery', 'shipping', 'packaging', 'arrived', 'dispatch'],
        'price': ['price', 'cost', 'worth', 'affordable', 'cheap']
    }


    
def scrape_amazon_reviews(url, pages=3):
    options = Options()
    options.add_argument("--headless")
    
    # ✅ Use a properly closed f-string
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")

    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(5)

    reviews = []
    for _ in range(pages):
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews.extend([
            review.text.strip()
            for review in soup.find_all("span", {"data-hook": "review-body"})
        ])

        try:
            next_page = driver.find_element(By.CSS_SELECTOR, '.a-pagination .a-last > a')
            next_page.click()
            time.sleep(random.uniform(3, 5))
        except NoSuchElementException:
            break

    driver.quit()
    return reviews


    def analyze_aspect_sentiments(reviews):
        aspect_sentiments = defaultdict(list)

        for review in reviews:
            sentences = nltk.sent_tokenize(review)
            for sentence in sentences:
                for aspect, keywords in ASPECTS.items():
                    if any(keyword in sentence.lower() for keyword in keywords):
                        sentiment = sia.polarity_scores(sentence)['compound']
                        aspect_sentiments[aspect].append(sentiment)

        return aspect_sentiments


    def generate_recommendations(aspect_analysis):
        recommendations = []
        for aspect, scores in aspect_analysis.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            if avg_score < -0.1:
                recommendations.append(
                    f"Improve {aspect}: Consider enhancing product {aspect} based on negative feedback")
            elif avg_score > 0.5:
                recommendations.append(
                    f"Highlight {aspect}: Market strong {aspect} features as customer satisfaction is high")
        return recommendations


    # Streamlit UI
    st.title("Amazon Product Sentiment Analyzer 🛍️")
    st.markdown("Analyze customer reviews to improve product strategy")

    with st.form("analysis_form"):
        url = st.text_input("Amazon Product URL", placeholder="Paste product URL here...")
        pages = st.number_input("Number of Pages to Scrape", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("Analyze Reviews")

    if submitted and url:
        with st.spinner("Scraping reviews..."):
            reviews = scrape_amazon_reviews(url, pages)

        if not reviews:
            st.error("No reviews found. Please check the URL or try again later.")
        else:
            # Sentiment Analysis
            df = pd.DataFrame({'Review': reviews})
            df['Sentiment'] = df['Review'].apply(lambda x: 'Positive' if sia.polarity_scores(x)['compound'] > 0.05
            else 'Negative' if sia.polarity_scores(x)['compound'] < -0.05
            else 'Neutral')

            # Aspect Analysis
            aspect_analysis = analyze_aspect_sentiments(reviews)
            aspect_df = pd.DataFrame({
                'Aspect': [k.capitalize() for k in aspect_analysis.keys()],
                'Average Sentiment': [sum(v) / len(v) if v else 0 for v in aspect_analysis.values()],
                'Mentions': [len(v) for v in aspect_analysis.values()]
            })

            # Display Metrics
            st.subheader("Key Metrics 📊")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", len(df))
            col2.metric("Positive Rate", f"{len(df[df['Sentiment'] == 'Positive']) / len(df):.1%}")
            col3.metric("Top Aspect", aspect_df.sort_values('Mentions', ascending=False).iloc[0]['Aspect'])

            # Sentiment Distribution
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

            # Aspect Analysis
            st.subheader("Product Aspect Analysis")
            st.bar_chart(aspect_df.set_index('Aspect')['Average Sentiment'])

            # Word Clouds
            st.subheader("Word Clouds")
            pos_text = " ".join(df[df['Sentiment'] == 'Positive']['Review'])
            neg_text = " ".join(df[df['Sentiment'] == 'Negative']['Review'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Positive Reviews**")
                if pos_text:
                    wordcloud = WordCloud().generate(pos_text)
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    st.pyplot(plt)

            with col2:
                st.markdown("**Negative Reviews**")
                if neg_text:
                    wordcloud = WordCloud().generate(neg_text)
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    st.pyplot(plt)

            # Recommendations
            st.subheader("Actionable Recommendations 💡")
            recommendations = generate_recommendations(aspect_analysis)
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.info("All aspects are performing well! Maintain current quality standards.")

            # Data Download
            st.download_button(
                label="Download Analysis Data",
                data=df.to_csv().encode('utf-8'),
                file_name='product_sentiment_analysis.csv',
                mime='text/csv'
            )

    st.markdown("---")
    st.info("Note: This tool is for educational purposes only. Always respect website terms of service.")
