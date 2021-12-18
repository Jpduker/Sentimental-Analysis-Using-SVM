# importing core packages
import streamlit as st

# Importing EDA packages
import pandas as pd
import numpy as np

import joblib

# function to connect with the model


def main():
    menu = ["Home", "monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Emotion Analysis App")

        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1, col2 = st.columns(2)

            # applying the functions

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")

            with col2:
                st.success("Prediction Probability")

    elif choice == "Monitor":
        st.subheader("Monitor-App")

    else:
        st.subheader("About")


if __name__ == "__main__":
    main()

