
import streamlit as st
import joblib
import re

# PAGE SETTINGS
st.set_page_config(
    page_title="ReviewGuard AI",
    page_icon="🔍",
    layout="centered"
)

# LOAD MODEL
model = joblib.load("fake_review_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# CUSTOM DESIGN
st.markdown("""
<style>
.stApp {
    background-color: #0C1424;
    color: white;
}

.block-container {
    padding-top: 3rem;
    max-width: 850px;
}

h1, h2, h3 {
    color: #FFFFFF !important;
}

.stButton > button {
    background-color: #3B82F6;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #2563EB;
    color: white;
}

.stTextArea textarea {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("🔍 ReviewGuard AI")

st.write(
    "### Detect Suspicious Reviews with AI"
)

st.caption(
    "Powered by Semi-Supervised PU Learning, "
    "TF-IDF and Logistic Regression"
)

st.divider()

# REVIEW INPUT
st.subheader("Analyze a Product Review")

review = st.text_area(
    "Enter your review:",
    placeholder="Paste a product review here...",
    height=160
)

# PREDICTION
if st.button("🔍 Analyze Review"):

    if review.strip():

        # Clean input
        cleaned = review.lower()
        cleaned = re.sub(
            r'[^a-z\s]', ' ', cleaned
        )
        cleaned = re.sub(
            r'\s+', ' ', cleaned
        ).strip()

        if cleaned:

            # TF-IDF transformation
            review_vector = tfidf.transform(
                [cleaned]
            )

            # Model prediction
            prediction = model.predict(
                review_vector
            )[0]

            # Display result
            st.divider()
            st.subheader("Analysis Result")

            if prediction == 1:
                st.error(
                    "⚠️ Potentially Fake Review"
                )
            else:
                st.success(
                    "✅ Likely Genuine Review"
                )

            st.info(
                "This is an AI prediction, "
                "not a verified fact."
            )

        else:
            st.warning(
                "Please enter valid review text."
            )

    else:
        st.warning(
            "Please enter a review first."
        )

# ABOUT SECTION
st.divider()

st.subheader("About the Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🧠 PU Learning")

with col2:
    st.info("📝 TF-IDF")

with col3:
    st.info("📊 Logistic Regression")

st.caption(
    "University Machine Learning Project | "
    "Fake Review Detection System"
)
