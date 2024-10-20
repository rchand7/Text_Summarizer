import streamlit as st
from transformers import pipeline
import os

# Set up the summarization models
MODEL_OPTIONS = {
    "BART": "facebook/bart-large-cnn",
    "T5": "t5-small",
    "Pegasus": "google/pegasus-xsum"
}

# Load the selected summarization model
def load_model(model_name):
    return pipeline("summarization", model=model_name)

# Streamlit app layout and features
st.set_page_config(page_title="Text Summarization App", layout="wide")

# App Title
st.title("Enhanced Text Summarization App")
st.write("Summarize text using different models and customize the output!")

# Model Selection
model_choice = st.sidebar.selectbox("Select Summarization Model", list(MODEL_OPTIONS.keys()))
model = load_model(MODEL_OPTIONS[model_choice])

# File Upload Section
st.subheader("Upload a text file or manually input text")

# File uploader for text files
uploaded_file = st.file_uploader("Choose a text file", type="txt")

# Input text area
input_text = st.text_area("Or enter text to summarize manually:", height=250)

# Parameters for controlling summary length
max_length = st.sidebar.slider("Maximum summary length:", 50, 300, 130)
min_length = st.sidebar.slider("Minimum summary length:", 20, 100, 30)

# Extract the text from the uploaded file, if available
if uploaded_file is not None:
    input_text = uploaded_file.read().decode("utf-8")
    st.write(f"**Word Count (Uploaded Text):** {len(input_text.split())} words")
    st.write(input_text)

# Show word count for manually entered input text
elif input_text:
    st.write(f"**Word Count (Input Text):** {len(input_text.split())} words")

# Button to trigger summarization
if st.button("Summarize"):
    if input_text:
        # Generate the summary
        summary = model(input_text, max_length=max_length, min_length=min_length, do_sample=False)
        summarized_text = summary[0]['summary_text']

        # Show the summary
        st.subheader("Summary:")
        st.write(summarized_text)

        # Show word count for the summary
        st.write(f"**Word Count (Summary):** {len(summarized_text.split())} words")

        # Option to save the summary as a text file
        if st.button("Download Summary"):
            with open("summary.txt", "w") as f:
                f.write(summarized_text)
            st.download_button(label="Click to download summary", data=summarized_text, file_name="summary.txt")

    else:
        st.write("Please upload a text file or enter some text to summarize.")

# Dark/Light Mode Toggle
theme_toggle = st.sidebar.checkbox("Dark Mode")
if theme_toggle:
    st.write('<style>body{background-color:#2E2E2E; color:white;}</style>', unsafe_allow_html=True)
else:
    st.write('<style>body{background-color:white; color:black;}</style>', unsafe_allow_html=True)

# Footer
st.write("---")
st.write("Developed by Rohit Chand")

  