import streamlit as st
import os

# Define the default PDF path
default_pdf_path = 'backend/data/pdfs/General_info_etc_2024.pdf'

# Function to scan and display the file
def scan_pdf(file_path):
    st.write(f"Scanning PDF at: {file_path}")
    # Your PDF processing code goes here

# Streamlit UI for file selection
st.title("PDF Scanner")

# File uploader to choose a file from the computer
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Button to use the default PDF path if no file is uploaded
if st.button("Use default PDF"):
    scan_pdf(default_pdf_path)
elif uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_path = os.path.join('data/pdfs/', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved file: {file_path}")
    scan_pdf(file_path)
