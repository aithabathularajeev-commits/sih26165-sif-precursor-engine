import streamlit as st

st.title("Input Test")

text = st.text_area(
    "Type something here:",
    height=200
)

st.write("You typed:", text)