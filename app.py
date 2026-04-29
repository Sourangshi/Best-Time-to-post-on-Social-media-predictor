import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os      # Fixed: Missing import
import gdown   # Fixed: Missing import
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

# --- Google Drive Model Config ---
MODEL_PATH = 'rf_model.pkl'
GOOGLE_DRIVE_FILE_ID = '1-F1oabEGcrJlf76KniVPLSvmErd8MM3Z' 
DOWNLOAD_URL = f'https://drive.google.com/drive/folders/1-F1oabEGcrJlf76KniVPLSvmErd8MM3Z?usp=drive_link'

@st.cache_resource
def load_rf_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... Please wait."):
            try:
                gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
    return joblib.load(MODEL_PATH)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analytics Dashboard", "Engagement Predictor"])

# ---------------------------------
# PAGE 1: ANALYTICS DASHBOARD
# ---------------------------------
if page == "Analytics Dashboard":
    st.title("Social Media Best Time To Post Optimizer")
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
        yt_rows = []
        try:
            youtube = build('youtube', 'v3', developerKey=API_KEY)
            CHANNEL_ID = "UC_x5XG1OV2P6uZZ5FSM9Ttw"
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
        except:
            pass

        yt = pd.DataFrame(yt_rows)
        df = pd.concat([insta, yt], ignore_index=True).dropna(subset=['publishedAt'])

        # --- FIXED: Preprocessing logic to prevent AttributeError ---
        for col in ['likes', 'comments', 'views', 'engagement_rate']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0)
        
        df['hour'] = df['publishedAt'].dt.hour
        df['day'] = df['publishedAt'].dt.dayofweek

        # Engagement Logic
        df['engagement'] = np.where(
            df['platform'] == 'Instagram',
            df['engagement_rate'],
            (df['likes'] + 2 * df['comments']) / (df['views'] / 1000 + 1)
        )

        # Scaling
        scaler = MinMaxScaler()
        df['engagement_scaled'] = 0.0
        for platform in df['platform'].unique():
            mask = df['platform'] == platform
            if mask.any():
                df.loc[mask, 'engagement_scaled'] = scaler.fit_transform(df.loc[mask, ['engagement']])
        return df

    df = load_data()
    z = zscore(df['engagement_scaled'])

    # Metrics (Kept exactly in your original location)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Posts", len(df))
    m2.metric("Avg Scaled Eng.", round(df['engagement_scaled'].mean(), 4))
    best_h = df.groupby('hour')['engagement_scaled'].mean().idxmax()
    m3.metric("Peak Hour", f"{best_h}:00")
    m4.metric("Anomalies", int((abs(z) > 2).sum()))

    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "⏰ Recommendations", "📉 Anomalies", "📂 Raw Data"])

    with tab1:
        p_choice = st.selectbox("Select Platform", df['platform'].unique())
        temp = df[df['platform'] == p_choice]
        pivot = temp.pivot_table(values='engagement_scaled', index='day', columns='hour', aggfunc='mean').fillna(0)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Best Time Recommendations")
        days_names=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        results=[
            {'platform':'Instagram', 'day':6, 'hour':13},
            {'platform':'YouTube', 'day':2, 'hour':23}
        ]
        for r in results:
            start, end = (r['hour']-1)%24, (r['hour']+1)%24
            st.success(
                f"**Platform: {r['platform']}**\n\n"
                f"Best Day: {days_names[r['day']]}\n\n"
                f"Best Hour: {r['hour']}:00\n\n"
                f"Suggested Window: {start}:00 to {end}:00"
            )
    
    with tab3:
        st.subheader("Engagement Anomaly Detection")
        df['anomaly'] = abs(z) > 2
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['engagement_scaled'], alpha=0.4, color='gray', label='Engagement')
        anom_df = df[df['anomaly']]
        ax.scatter(anom_df.index, anom_df['engagement_scaled'], color='red', label='Anomaly')
        ax.legend()
        st.pyplot(fig)

    with tab4:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(50))

# ---------------------------------
# PAGE 2: PREDICTOR
# ---------------------------------
elif page == "Engagement Predictor":
    st.title("🔮 Best Time To Post Predictor")
    st.caption("Predict engagement score based on your Random Forest model")

    try:
        rf = load_rf_model()
        col_a, col_b = st.columns(2)
        with col_a:
            hour = st.slider("Posting Hour", 0, 23, 18)
            day = st.slider("Day (0=Mon ... 6=Sun)", 0, 6, 2)
            followers = st.number_input("Followers", value=10000)
        with col_b:
            caption_length = st.number_input("Caption Length", value=100)
            hashtags = st.number_input("Hashtags", value=5)

        if st.button("Predict Engagement"):
            media_type, content_cat, traffic, cta = 1, 1, 1, 1
            row = [[hour, day, followers, caption_length, hashtags, media_type, content_cat, traffic, cta]]
            prediction = rf.predict(row)
            st.metric("Predicted Engagement Score", round(prediction[0], 4))
            
    except Exception as e:
        st.error(f"Error: {e}")
