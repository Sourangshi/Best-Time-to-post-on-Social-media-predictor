import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from googleapiclient.discovery import build
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Social Media Optimizer & Predictor",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 Analytics Dashboard", "🔮 Engagement Predictor"])

# ---------------------------------
# PAGE 1: ANALYTICS DASHBOARD
# ---------------------------------
if page == "📈 Analytics Dashboard":
    st.title("📊 Social Media Best Time To Post Optimizer")
    st.caption("Instagram + YouTube Analytics Dashboard")

    @st.cache_data
    def load_data():
        # --- Instagram Data ---
        try:
            insta = pd.read_csv("Instagram_Analytics.csv")
            insta = insta.loc[:, ~insta.columns.duplicated()].copy()
            insta['platform'] = 'Instagram'
            if 'post_datetime' in insta.columns:
                insta['publishedAt'] = pd.to_datetime(insta['post_datetime'], errors='coerce', utc=True)
        except:
            insta = pd.DataFrame()

        # --- YouTube API ---
        API_KEY = "AIzaSyDlTb2GI5yRCzrm1mnPS0OFGdIlABje-xo"
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        CHANNEL_ID = "UC_x5XG1OV2P6uZZ5FSM9Ttw"

        yt_rows = []
        try:
            ch_res = youtube.channels().list(part="contentDetails", id=CHANNEL_ID).execute()
            playlist_id = ch_res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            pl_res = youtube.playlistItems().list(part="contentDetails", playlistId=playlist_id, maxResults=50).execute()
            video_ids = [item['contentDetails']['videoId'] for item in pl_res['items']]
            if video_ids:
                stats_res = youtube.videos().list(part="statistics,snippet", id=",".join(video_ids)).execute()
                for item in stats_res['items']:
                    s = item['statistics']
                    yt_rows.append({
                        'platform': 'YouTube',
                        'publishedAt': pd.to_datetime(item['snippet']['publishedAt'], utc=True),
                        'likes': int(s.get('likeCount', 0)),
                        'comments': int(s.get('commentCount', 0)),
                        'views': int(s.get('viewCount', 0))
                    })
        except Exception as e:
            st.sidebar.warning("YouTube API limit reached or Error.")

        yt = pd.DataFrame(yt_rows)
        
        # Merge
        df = pd.concat([insta, yt], ignore_index=True)
        df = df.dropna(subset=['publishedAt'])

        # Preprocessing & Scoring (Matching Project Logic)
        df['likes'] = df['likes'].fillna(0)
        df['comments'] = df['comments'].fillna(0)
        df['views'] = df['views'].fillna(0)
        df['engagement_rate'] = df.get('engagement_rate', 0).fillna(0)
        df['hour'] = df['publishedAt'].dt.hour
        df['day'] = df['publishedAt'].dt.dayofweek

        df['engagement'] = np.where(
            df['platform'] == 'Instagram',
            df['engagement_rate'],
            (df['likes'] + 2 * df['comments']) / (df['views'] / 1000 + 1)
        )

        # Scaling
        df['engagement_scaled'] = 0.0
        scaler = MinMaxScaler()
        for platform in df['platform'].unique():
            mask = df['platform'] == platform
            if mask.any():
                df.loc[mask, 'engagement_scaled'] = scaler.fit_transform(df.loc[mask, ['engagement']])
        return df

    df = load_data()

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Posts", len(df))
    m2.metric("Avg Scaled Eng.", round(df['engagement_scaled'].mean(), 4))
    best_h = df.groupby('hour')['engagement_scaled'].mean().idxmax()
    m3.metric("Peak Hour", f"{best_h}:00")
    z = zscore(df['engagement_scaled'])
    m4.metric("Anomalies", int((abs(z) > 2).sum()))

    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "⏰ Recommendations", "📉 Anomalies", "📂 Data"])

    with tab1:
        p_choice = st.selectbox("Select Platform", df['platform'].unique())
        temp = df[df['platform'] == p_choice]
        pivot = temp.pivot_table(values='engagement_scaled', index='day', columns='hour', aggfunc='mean').fillna(0)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)


    with tab2:

      st.subheader(
      "Best Time Recommendations"
    )

    days=[
      'Mon','Tue','Wed',
      'Thu','Fri','Sat','Sun'
    ]

    results=[
        {
          'platform':'Instagram',
          'day':6,
          'hour':13
        },
        {
          'platform':'YouTube',
          'day':2,
          'hour':23
        }
     ]

    for r in results:

        start=(r['hour']-1)%24
        end=(r['hour']+1)%24

        st.success(
            f"Platform: {r['platform']}\n\n"
            f"Best Day: {days[r['day']]}\n\n"
            f"Best Hour: {r['hour']}:00\n\n"
            f"Suggested Window: {start}:00 to {end}:00"
        )
    
    with tab3:
        df['anomaly'] = abs(z) > 2
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['engagement_scaled'], alpha=0.4, color='gray')
        ax.scatter(df[df['anomaly']].index, df[df['anomaly']]['engagement_scaled'], color='red')
        st.pyplot(fig)

    with tab4:
        st.dataframe(df.head(50))

# ---------------------------------
# PAGE 2: PREDICTOR (Your Original ML Code)
# ---------------------------------
elif page == "🔮 Engagement Predictor":
    st.title("🔮 Best Time To Post Predictor")
    st.caption("Predict engagement for a specific post using your Random Forest Model")

    try:
        rf = joblib.load("rf_model.pkl")
        
        col_a, col_b = st.columns(2)
        with col_a:
            hour = st.slider("Posting Hour", 0, 23, 18)
            day = st.slider("Day (0=Mon ... 6=Sun)", 0, 6, 2)
            followers = st.number_input("Followers", value=10000)
        with col_b:
            caption_length = st.number_input("Caption Length", value=100)
            hashtags = st.number_input("Hashtags", value=5)

        if st.button("Predict Engagement"):
            # Missing features for your 9-feature model
            media_type, content_cat, traffic, cta = 1, 1, 1, 1
            row = [[hour, day, followers, caption_length, hashtags, media_type, content_cat, traffic, cta]]
            prediction = rf.predict(row)
            st.metric("Predicted Engagement Score", round(prediction[0], 4))
            
    except Exception as e:
        st.error(f"Could not load model 'rf_model.pkl'. Please ensure it is in the same folder. Error: {e}")