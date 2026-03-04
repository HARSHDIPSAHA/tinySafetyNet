import streamlit as st

st.title("Test")

audio = st.audio_input("Record something")

if audio:
    st.write("Audio received")
