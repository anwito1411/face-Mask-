# app.py
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(page_title="Mask Detector", page_icon="😷")
st.title("😷 AI Face Mask Detector")
st.write("Upload an image to check if people are wearing masks.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze Image'):
        with st.spinner('Analyzing...'):
            # Send to FastAPI
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            files = {"file": ("image.png", img_byte_arr, "image/png")}
            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                data = response.json()
                
                # Draw boxes
                draw = ImageDraw.Draw(image)
                # Try to load a font, fallback to default if not found
                try:
                    font = ImageFont.truetype("arial.ttf", size=20)
                except:
                    font = ImageFont.load_default()

                if not data["predictions"]:
                    st.warning("No faces detected!")
                else:
                    for pred in data["predictions"]:
                        (x, y, w, h) = pred["bbox"]
                        label = pred["label"]
                        conf = pred["confidence"]
                        
                        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
                        
                        # Draw Box
                        draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=3)
                        # Draw Label
                        text = f"{label}: {conf:.2f}%"
                        draw.text((x, y - 25), text, fill=color, font=font)
                    
                    st.success(f"Found {len(data['predictions'])} faces.")
                    st.image(image, caption='Processed Image', use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
