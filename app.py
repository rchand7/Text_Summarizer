import streamlit as st
from transformers import pipeline

# Set up the summarization models
MODEL_OPTIONS = {
    "BART (Base)": "facebook/bart-base",
    "T5 (Small)": "t5-small",
    "Pegasus (XSUM)": "google/pegasus-xsum"
}

# Load the selected summarization model
@st.cache_resource
def load_model(model_name):
    return pipeline("summarization", model=model_name, device=-1)  # Force CPU usage

# Streamlit app layout and features
st.set_page_config(page_title="Enhanced Text Summarization App", layout="wide")

# App Title
st.title("Enhanced Text Summarization App")
st.write("Summarize text using different models and customize the output!")

# Model Selection
model_choice = st.sidebar.selectbox("Select Summarization Model", list(MODEL_OPTIONS.keys()))
model = load_model(MODEL_OPTIONS[model_choice])

# Input text area
input_text = st.text_area("Enter text to summarize:", height=250)

# Parameters for controlling summary length
max_length = st.sidebar.slider("Maximum summary length:", 50, 800, 350)
min_length = st.sidebar.slider("Minimum summary length:", 20, 300, 130)

# Show word count for input text
if input_text:
    st.write(f"**Word Count (Input):** {len(input_text.split())} words")

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
        st.download_button(label="Click to download summary", data=summarized_text, file_name="summary.txt", mime="text/plain")

    else:
        st.write("Please enter some text to summarize.")

# Dark/Light Mode Toggle
theme_toggle = st.sidebar.checkbox("Dark Mode")
if theme_toggle:
    st.write('<style>body{background-color:#2E2E2E; color:white;}</style>', unsafe_allow_html=True)
else:
    st.write('<style>body{background-color:white; color:black;}</style>', unsafe_allow_html=True)

# Footer
st.write("---")
st.write("Developed by Rohit Chand")
