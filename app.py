import streamlit as st
import main
from main import sentiment_analyzer

st.set_page_config(layout="wide")


st.markdown(
    """
    <h1 style='text-align: center; color: #FFD700;'>Movie Review Sentiment Analyzer</h1>
    """,
    unsafe_allow_html=True
)


user_input = st.text_input("User", help="Enter your message here")

if st.button("Send"):
    st.text_area("User Review", value=user_input, height=200)

    sentiment = sentiment_analyzer(user_input)
    st.text_area("Sentiment", value=sentiment, height=30)
