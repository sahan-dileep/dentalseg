import streamlit as st
from ultralytics import YOLO
import cv2

st.title("Dental Segmentation App")

# Load the YOLO model
model = YOLO("weight/best.pt")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run inference
    results = model("uploaded_image.jpg")

    # Save the result
    for result in results:
        result.save(filename="result.jpg")

    # Display the result
    st.image("result.jpg", caption="Inference Result", use_column_width=True)