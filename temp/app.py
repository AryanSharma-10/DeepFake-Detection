import streamlit as st
import tempfile
from model import load_model
from video_processing import process_video

# Load the trained ResNet50 model
model = load_model('resnet50_new_model_v2.h5')

# Set the title of the app
st.title("DeepFake Detection")

# Add a file uploader for video files
uploaded_video = st.file_uploader("Choose a video to upload", type=["mp4", "avi", "mov"])

# Display the uploaded video and process it
if uploaded_video is not None:
    name = uploaded_video.name
    # Clear the page
    st.empty()

    with st.spinner("Analyzing video..."):
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        # Process video
        output_video_path = "../Streamlit/annotated_video.mp4"
        deepfake_percentage = process_video(tfile.name, model, output_video_path, name)

    # Display the results
    st.video(output_video_path)
    st.write(f"DeepFake frames detected: {deepfake_percentage:.2f}%")
else:
    st.write("No video uploaded.")
