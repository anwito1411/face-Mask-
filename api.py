from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI()

# Load the Mask Detector Model
print("[INFO] Loading mask detector model...")
model = load_model("mask_detector.h5")

# Load OpenCV's built-in Face Detector (Haar Cascade)
# This works on ALL Python versions without extra installs
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read and Convert Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    
    # Convert RGB to Grayscale (OpenCV face detector prefers gray)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 2. Detect Faces
    # scaleFactor=1.1, minNeighbors=5 are standard tuning params
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    predictions = []
    
    # 3. Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract the Face (Region of Interest)
        face_roi = image_np[y:y+h, x:x+w]
        
        # Preprocess for MobileNet (Resize to 224x224)
        try:
            face_input = cv2.resize(face_roi, (224, 224))
            face_input = img_to_array(face_input)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            # Predict Mask vs No Mask
            (mask, withoutMask) = model.predict(face_input)[0]
            
            label = "Mask" if mask > withoutMask else "No Mask"
            prob = max(mask, withoutMask) * 100
            
            predictions.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "label": label,
                "confidence": float(prob)
            })
        except Exception as e:
            print(f"Skipping a face due to error: {e}")
            continue
            
    return {"predictions": predictions}

# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
