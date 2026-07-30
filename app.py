# app.py
# ==========================================================
# Indian Government Schemes Personalized Chatbot
# Streamlit App with Evaluation Metrics
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string 
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import precision_score, recall_score, f1_score    

# ----------------------------------------------------------  
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Gov Scheme Chatbot",
    page_icon="🇮🇳",
    layout="wide"
)

# ----------------------------------------------------------
# NLTK DOWNLOADS
# ----------------------------------------------------------
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# ----------------------------------------------------------
# LOAD MODEL FILES
# ----------------------------------------------------------
@st.cache_resource
def load_files():
    tfidf = joblib.load("tfidf.pkl")
    scheme_matrix = joblib.load("scheme_matrix.pkl")
    df = joblib.load("cleaned_dataset.pkl")
    return tfidf, scheme_matrix, df

tfidf, scheme_matrix, df = load_files()

# ----------------------------------------------------------
# CLEAN TEXT
# ----------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ----------------------------------------------------------
# PROFILE EXTRACTION
# ----------------------------------------------------------
def extract_profile(query):

    q = query.lower()

    return {
        "student": int("student" in q),
        "farmer": int("farmer" in q),
        "woman": int("female" in q or "woman" in q or "girl" in q),
        "startup": int("startup" in q or "business" in q or "entrepreneur" in q),
        "disabled": int("disabled" in q or "disability" in q),
        "scholarship": int("scholarship" in q or "study" in q or "education" in q)
    }

# ----------------------------------------------------------
# ELIGIBILITY SCORE
# ----------------------------------------------------------
def eligibility_score(profile, text):

    score = 0

    if profile["student"] and "student" in text:
        score += 1

    if profile["farmer"] and "farmer" in text:
        score += 1

    if profile["woman"] and ("women" in text or "female" in text):
        score += 1

    if profile["startup"] and ("startup" in text or "business" in text):
        score += 1

    if profile["disabled"] and ("disabled" in text or "disability" in text):
        score += 1

    if profile["scholarship"] and ("scholarship" in text or "education" in text):
        score += 1

    return score / 6

# ----------------------------------------------------------
# RECOMMEND FUNCTION
# ----------------------------------------------------------
def recommend(query, top_n=5):

    processed = clean_text(query)

    user_vec = tfidf.transform([processed])

    similarity = cosine_similarity(user_vec, scheme_matrix).flatten()

    profile = extract_profile(query)

    scores = []

    for i in range(len(df)):
        txt = df.iloc[i]["processed_text"]
        elig = eligibility_score(profile, txt)

        final = (0.70 * similarity[i]) + (0.30 * elig)
        scores.append(final)

    temp = df.copy()
    temp["score"] = scores

    result = temp.sort_values("score", ascending=False).head(top_n)

    return result

# ----------------------------------------------------------
# MODEL EVALUATION
# ----------------------------------------------------------
def evaluate_model():

    test_queries = [
        ("female student scholarship", ["scholarship", "education"]),
        ("farmer subsidy", ["farmer", "agriculture"]),
        ("startup loan", ["startup", "business"]),
        ("disabled pension", ["disabled", "pension"]),
        ("women self employment", ["women", "loan"])
    ]

    y_true = []    
    y_pred = []                 

    for query, keywords in test_queries:                       

        result = recommend(query)

        for _, row in result.iterrows():

            text = (
                str(row["scheme_name"]).lower() + " " +
                str(row["benefits"]).lower() + " " +
                str(row["eligibility"]).lower()
            )

            matched = any(word in text for word in keywords)

            y_true.append(1)
            y_pred.append(1 if matched else 0)

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, precision, recall, f1

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("⚙️ NLP Pipeline")

st.sidebar.markdown("""
### Steps Used

1. Lowercase Conversion  
2. Remove Numbers  
3. Remove Punctuation  
4. Stopword Removal  
5. TF-IDF Vectorization  
6. Cosine Similarity  
7. Eligibility Matching  
8. Top 5 Ranking
""")

st.sidebar.markdown("---")
st.sidebar.title("📊 Dataset Info")
st.sidebar.write("Total Schemes:", len(df))

# Evaluation
acc, precision, recall, f1 = evaluate_model()

st.sidebar.markdown("---")
st.sidebar.title("📈 Evaluation Scores")

st.sidebar.write("Accuracy:", round(acc,3))
st.sidebar.write("Precision:", round(precision,3))
st.sidebar.write("Recall:", round(recall,3))
st.sidebar.write("F1 Score:", round(f1,3))

# ----------------------------------------------------------
# MAIN TITLE
# ----------------------------------------------------------
st.title("🇮🇳 Indian Government Schemes Chatbot")
st.write("Enter your profile and get top personalized schemes.")

query = st.text_area(
    "Example: I am a female student from Haryana with family income 2 lakh and need scholarship",
    height=130
)

# ----------------------------------------------------------
# BUTTON
# ----------------------------------------------------------
if st.button("Find Schemes"):

    if query.strip() == "":
        st.warning("Please enter your details.")
        st.stop()

    result = recommend(query)

    st.success("Top Personalized Schemes Found")

    for _, row in result.iterrows():

        with st.expander(f"🏆 {row['scheme_name']}"):

            st.write("### Match Score")
            st.progress(float(min(row["score"],1.0)))

            st.write("### Benefits")
            st.write(row["benefits"])

            st.write("### Eligibility")
            st.write(row["eligibility"])

            st.write("### Application")
            st.write(row["application"])

    # ------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------
    st.subheader("📈 Recommendation Confidence")

    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(x=result["scheme_name"], y=result["score"], ax=ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------------------------------------
# EVALUATION GRAPH
# ----------------------------------------------------------
st.subheader("📊 Model Evaluation Dashboard")

fig2, ax2 = plt.subplots(figsize=(8,4))
sns.barplot(
    x=["Accuracy","Precision","Recall","F1"],
    y=[acc, precision, recall, f1],
    ax=ax2
)
plt.ylim(0,1)
plt.tight_layout()
st.pyplot(fig2)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.caption("Built with Python + Streamlit + NLP + TF-IDF")