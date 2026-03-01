import streamlit as st
import tempfile
import os
from src.stream_processor import process_video

st.title("Real-Time Person Detection & Bi-Directional Counting")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    output_path = "processed_output.mp4"

    st.write("Processing video... Please wait.")

    process_video(input_path, output_path)

    st.success("Processing complete!")

    st.video(output_path)