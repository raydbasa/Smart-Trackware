#Import List
import os
import cv2
import json
import subprocess
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import base64  
from PIL import Image
from io import BytesIO
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
import uuid
from model_processing import load_model, image_processing, video_processing

# Streamlit page configuration
st.set_page_config(page_title="Smart Track Ware", layout="wide", page_icon="./SmarT.ico")
st.markdown(
    """
    <style>
       .stMarkdown {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='color: white;'>Smart Track Ware</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
       .stMarkdown {
        color: white;
        text-align: center; /* Center-align the text */
        margin: auto; /* Center the text horizontally */
        max-width: 800px; /* Set a maximum width for the text */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .white-subheader {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Project Description
st.write("""
    **Revolutionize Your Warehouse with AI-Powered Inventory Management!**

    Designed to automate and optimize inventory management, our state-of-the-art system ensures unprecedented efficiency, minimizes errors. Experience precise counting of inventory items, providing you with accurate, 
    up-to-the-minute insights for seamless inventory control.""")

def add_bg_from_local(image_file, opacity=0.5):
    with open(image_file, "rb") as f:
        image = Image.open(f)
        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, int(255 * opacity)))
        image = Image.alpha_composite(image, overlay)
        image_with_bg = image.convert("RGB")

        # Encoding the image to base64
        buffered = BytesIO()
        image_with_bg.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{encoded_image}');
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to add background from the local image file
image_file_path = r"C:\Users\aljow\Desktop\Streamlit\box.png"
add_bg_from_local(image_file_path, opacity=0.6)  # Adjust opacity as desired

# Initialize directories and model
model_file = "best.pt"
model = load_model(model_file)

# Sidebar for file upload options
tab_upload = st.sidebar.radio("Upload", ["Image", "Video"])

if tab_upload == "Image":
    st.sidebar.header("Upload an image")
    image_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])
    process_image_button = st.sidebar.button("Process Image")

    if image_file is not None and process_image_button:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        img, result_list_json, object_count = image_processing(img, model)
        
        # Applying the CSS classes
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        st.markdown(f'<h3 class="white-subheader">Total Objects Counted: {object_count}</h3>', unsafe_allow_html=True)
        st.image(img, caption="Processed image", channels="BGR")
        st.markdown('</div>', unsafe_allow_html=True)

elif tab_upload == "Video":
    st.sidebar.header("Upload a video")
    video_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    process_video_button = st.sidebar.button("Process Video")

    if video_file is not None and process_video_button:
        filename = str(uuid.uuid4()) + ".mp4"
        tracker = DeepSort(max_age=5)
        centers = [deque(maxlen=30) for _ in range(10000)]
        
        with open(filename, "wb") as f:
            f.write(video_file.read())
        
        video_file_out, result_video_json_file, total_counted = video_processing(filename, model, tracker=tracker, centers=centers)
        
        if video_file_out is not None:
            st.markdown(f'<h3 class="white-subheader">Total Objects Counted: {total_counted}</h3>', unsafe_allow_html=True)
            video_bytes = open(video_file_out, 'rb').read()
            st.video(video_bytes)
        else:
            st.error("Failed to process the video.")
            os.remove(filename)
