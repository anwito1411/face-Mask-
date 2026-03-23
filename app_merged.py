import streamlit as st
import cv2
import keras
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.set_page_config(page_title="Mask Detector", page_icon="😷")
st.title("😷 Real-Time Face Mask Detector")

# 1. Load Model directly in Streamlit (Cache it so it's fast)
@st.cache_resource
def load_my_model():
    try:
        # We use keras.models.load_model directly for better compatibility
        return keras.models.load_model("mask_detector.keras", compile=False)
    except Exception as e:
        # This will show us the real error if it fails
        st.error(f"CRITICAL ERROR: {e}")
        return None

model = load_my_model()
if model is None:
    st.stop() # Stop the app if model fails
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    if st.button('Analyze'):
        # Process image
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            face_roi = image_np[y:y+h, x:x+w]
            face_input = cv2.resize(face_roi, (224, 224))
            face_input = img_to_array(face_input)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            (mask, withoutMask) = model.predict(face_input)
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
            cv2.rectangle(image_np, (x, y), (x+w, y+h), color, 10)
            
        st.image(image_np, caption='Processed Image', use_container_width=True)
        st.success("Analysis Complete!")
