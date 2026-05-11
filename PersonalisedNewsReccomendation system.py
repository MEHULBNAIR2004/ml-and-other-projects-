import streamlit as st
import pandas as pd
import re
import requests
import xml.etree.ElementTree as ET
import numpy as np
from urllib.parse import quote
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    layout="wide",
    page_title="Personalised News Recommendation System"
)

# ==============================
# 1. ENHANCED AI MODELS WITH EVALUATION
# ==============================
@st.cache_resource
def load_models(data):
    if data.empty: return None, None, None
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = vec.fit_transform(data['cleaned_text'])
    bm25_model = BM25Okapi([doc.split() for doc in data['cleaned_text']])
    return vec, matrix, bm25_model

def evaluate_and_recommend(text, mode="Hybrid", top_n=5):
    if vectorizer is None: return [], []
    
    query = re.sub(r'[^a-z\s]', '', str(text).lower())
    query_tokens = query.split()
    
    # 1. Calculate TF-IDF Scores
    q_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    
    # 2. Calculate BM25 Scores
    bm25_raw_scores = np.array(bm25.get_scores(query_tokens))
    
    # 3. Normalization (Scaling to 0-1 for Evaluation Comparison)
    t_min, t_max = tfidf_scores.min(), tfidf_scores.max()
    tfidf_norm = (tfidf_scores - t_min) / (t_max - t_min + 1e-9)
    
    b_min, b_max = bm25_raw_scores.min(), bm25_raw_scores.max()
    bm25_norm = (bm25_raw_scores - b_min) / (b_max - b_min + 1e-9)
    
    # Selection Logic
    if mode == "TF-IDF Only":
        final_scores = tfidf_norm
    elif mode == "BM25 Only":
        final_scores = bm25_norm
    else: # Hybrid
        final_scores = (0.5 * tfidf_norm) + (0.5 * bm25_norm)
    
    indices = final_scores.argsort()[::-1][:top_n]
    scores = final_scores[indices]
    
    return indices, scores

# ==============================
# 2. DATA & UTILS
# ==============================
def get_live_news(query):
    try:
        encoded = quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        return [{'title': i.find('title').text, 'link': i.find('link').text} for i in root.findall('.//item')[:5]]
    except: return []

@st.cache_data
def load_data():
    df = pd.read_csv("BBC_news.csv")
    df = df.drop_duplicates(subset='title').reset_index(drop=True)
    df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
    df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r'[^a-z\s]', '', str(x).lower()))
    return df

df = load_data()
vectorizer, tfidf_matrix, bm25 = load_models(df)

# ==============================
# 3. SESSION STATE
# ==============================
if "active_article" not in st.session_state: st.session_state.active_article = None
if "history_text" not in st.session_state: st.session_state.history_text = ""
if "last_query" not in st.session_state: st.session_state.last_query = ""

def handle_click(title, desc=""):
    st.session_state.active_article = {"title": title, "desc": desc}
    st.session_state.history_text += f" {title}"
    st.toast(f"Model Analyzing: {title[:20]}...")

# ==============================
# 4. SIDEBAR: MODEL CONTROL CENTER
# ==============================
with st.sidebar:
    st.title("⚙️ Model Control")
    
    model_choice = st.selectbox(
        "Choose Recommendation Model",
        ["Hybrid (TF-IDF + BM25)", "TF-IDF Only", "BM25 Only"]
    )
    
    st.markdown("---")
    st.subheader("📊 Evaluation Metrics")
    
    if st.session_state.active_article:
        # Calculate scores for evaluation display
        _, eval_scores = evaluate_and_recommend(st.session_state.active_article['title'], mode=model_choice)
        
        avg_score = np.mean(eval_scores) if len(eval_scores) > 0 else 0
        
        st.metric("Model Confidence", f"{avg_score:.2%}")
        st.caption("Mean similarity score of top recommendations.")
        
        if avg_score > 0.4:
            st.success("High Relevance Detected")
        elif avg_score > 0.15:
            st.warning("Moderate Relevance")
        else:
            st.error("Low Relevance / Niche Topic")
            
    if st.button("Clear Session"):
        st.session_state.active_article = None
        st.rerun()

# ==============================
# 5. MAIN UI
# ==============================
st.title("🌌 Personalised news Recommendation System")

query = st.text_input("Enter Topic:", placeholder="Search live & local...")

if query and query != st.session_state.last_query:
    st.session_state.last_query = query
    handle_click(query)
    st.rerun()

if st.session_state.active_article:
    active = st.session_state.active_article
    st.markdown(f"### 📍 Context: **{active['title']}**")
    
    l_col, r_col = st.columns(2)
    
    with l_col:
        st.markdown("#### 🌐 Live News")
        live_list = get_live_news(active['title'])
        for i, item in enumerate(live_list):
            if st.button(f"🔗 {item['title']}", key=f"live_{i}"):
                handle_click(item['title'])
                st.rerun()

    with r_col:
        st.markdown(f"#### 📜 Archives ({model_choice})")
        
        # Get recommendations and the mathematical scores
        indices, scores = evaluate_and_recommend(active['title'] + " " + active.get('desc', ''), mode=model_choice)
        
        for idx, score in zip(indices, scores):
            row = df.iloc[idx]
            with st.container():
                # Display individual Evaluation Score for each article
                st.write(f"**Score: `{score:.4f}`**") 
                if st.button(f"📖 {row['title']}", key=f"local_{idx}"):
                    handle_click(row['title'], row['description'])
                    st.rerun()
                st.caption(f"{row['description'][:100]}...")
                st.markdown("---")