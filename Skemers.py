# # importing libraries
# import streamlit as st
# import joblib

# #Loading the model 
# vectorizer, model = joblib.load("Logistic_model.pkl")

# # Creating the app
# st.write("# Twitter sentiment analysis app")
# st.write("## Hello, fill in a tweet and you will get a sentiment")


# def predict_sentiment(scenario):
#     # Clean and preprocess the input if needed
#     processed = vectorizer.transform([scenario])
#     prediction = model.predict(processed)
    
#     if prediction[0] == 1:
#         print('Positive')
#     elif prediction[0] == -1:
#         print('Negative')
#     else:
#         print('Neutral')

# #Sentiment outputs
# st.text_input("Type the Tweet below")
# scenario = st.text_input("Type the Tweet below")
# st.write(f"This tweet is: {predict_sentiment(scenario)}")

# importing libraries
import streamlit as st
import joblib

# Load vectorizer and model
vectorizer, model = joblib.load("Logistic_model.pkl")

# App UI
st.title("Twitter Sentiment Analysis App")
st.subheader("Enter a tweet to predict its sentiment")

# Input field
scenario = st.text_input("Type the Tweet below", key="tweet_input")

# Prediction function
def predict_sentiment(scenario):
    processed = vectorizer.transform([scenario])
    prediction = model.predict(processed)
    
    if prediction[0] == 1:
        return "Positive"
    elif prediction[0] == -1:
        return "Negative"
    else:
        return "Neutral"
# Button to trigger prediction
if st.button("Predict Sentiment"):
    if not scenario.strip():
        st.warning("Please enter a tweet.")
    else:
        result = predict_sentiment(scenario)
        st.write(f"This tweet is: {result}")
