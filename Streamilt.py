# app.py
import streamlit as st
import joblib

@st.cache_resource
def load_model(path="Logistic_model.pkl"):
    """
    Load model and vectorizer from disk once and cache them.
    """
    vectorizer, model = joblib.load(path)
    return vectorizer, model

def predict_sentiment(text: str, vectorizer, model) -> str:
    """
    Transform input text and predict sentiment label.
    """
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    if label == 1:
        return "Positive "
    elif label == -1:
        return "Negative "
    else:
        return "Neutral "

def main():
    st.title("💬 Twitter Sentiment Analyzer")
    st.subheader("Enter a tweet to predict its sentiment")

    vectorizer, model = load_model()
    st.info(f"Model loaded: {type(model).__name__}; Vect: {type(vectorizer).__name__}")

    tweet = st.text_input("Tweet text", key="tweet_input")
    
    if st.button("Analyze"):
        if not tweet or not tweet.strip():
            st.warning("Please enter a tweet.")
        else:
            result = predict_sentiment(tweet.strip(), vectorizer, model)
            st.success(f"Sentiment: **{result}**")

if __name__ == "__main__":
    main()
