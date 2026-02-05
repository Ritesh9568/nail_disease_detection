# test.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# Load Model
# =========================
MODEL_PATH = "model/nail_disease_model.h5"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# Image Path
# =========================
IMG_PATH = "dataset/test/test/leukonychia/19.PNG"

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"❌ Image not found: {IMG_PATH}")

# =========================
# Class Names (MUST match training order)
# =========================
class_names = [
    "Alopecia Areata",
    "Beau's Lines",
    "Bluish Nail",
    "Clubbing",
    "Darier Disease",
    "Eczema",
    "Half and Half Nails",
    "Koilonychia",
    "Leukonychia",
    "Muehrcke's Lines",
    "Onycholysis",
    "Pale Nail",
    "Red Lunula",
    "Splinter Hemorrhage",
    "Terry's Nail",
    "White Nail",
    "Yellow Nails"
]

# =========================
# Preprocess Image
# =========================
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

img_array = preprocess_image(IMG_PATH)

# =========================
# Prediction
# =========================
preds = model.predict(img_array)

predicted_index = np.argmax(preds)
confidence = round(float(np.max(preds)) * 100, 2)

# =========================
# Top-3 Predictions (Optional but Powerful)
# =========================
top_3_idx = preds[0].argsort()[-3:][::-1]

print("\n🩺 Prediction Results")
print("-" * 30)

for i, idx in enumerate(top_3_idx, start=1):
    print(
        f"{i}. {class_names[idx]} "
        f"({round(float(preds[0][idx]) * 100, 2)}%)"
    )

print("\n✅ Final Prediction:")
print("Disease   :", class_names[predicted_index])
print("Confidence:", confidence, "%")
