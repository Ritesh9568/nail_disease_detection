from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# =======================
# Flask App Init
# =======================
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =======================
# Load Trained Model
# =======================
MODEL_PATH = "model/nail_disease_model.h5"
model = load_model(MODEL_PATH)
print("✅ Nail disease model loaded successfully")


# =======================
# Class Names (MUST match training order)
# =======================
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


# =======================
# Disease Information
# =======================
disease_info = {
    "Alopecia Areata": {
        "description": "An autoimmune condition causing hair loss and nail pitting.",
        "symptoms": "Nail pitting, hair loss, brittle nails",
        "causes": "Immune system attacks hair follicles",
        "precautions": "Consult a dermatologist, reduce stress"
    },
    "Beau's Lines": {
        "description": "Horizontal grooves across fingernails.",
        "symptoms": "Indented lines on nails",
        "causes": "Severe illness, chemotherapy, trauma",
        "precautions": "Treat the underlying illness"
    },
    "Bluish Nail": {
        "description": "Blue discoloration due to poor oxygen supply.",
        "symptoms": "Blue or purple nail beds",
        "causes": "Heart or lung problems",
        "precautions": "Seek medical evaluation immediately"
    },
    "Clubbing": {
        "description": "Enlargement of fingertips and curved nails.",
        "symptoms": "Rounded nail tips",
        "causes": "Chronic lung or heart disease",
        "precautions": "Medical diagnosis required"
    },
    "Eczema": {
        "description": "Inflammatory skin condition affecting nails.",
        "symptoms": "Cracked, ridged nails",
        "causes": "Allergies, genetics",
        "precautions": "Moisturize and avoid irritants"
    },
    "Koilonychia": {
        "description": "Spoon-shaped nails.",
        "symptoms": "Thin, concave nails",
        "causes": "Iron deficiency anemia",
        "precautions": "Iron-rich diet and blood tests"
    },
    "Leukonychia": {
        "description": "White spots or lines on nails.",
        "symptoms": "White discoloration",
        "causes": "Minor trauma or zinc deficiency",
        "precautions": "Maintain a balanced diet"
    },
    "Onycholysis": {
        "description": "Separation of nail from the nail bed.",
        "symptoms": "White nail edges",
        "causes": "Infection or injury",
        "precautions": "Keep nails dry and short"
    },
    "Yellow Nails": {
        "description": "Yellow discoloration of nails.",
        "symptoms": "Thick, slow-growing nails",
        "causes": "Fungal infection or smoking",
        "precautions": "Stop smoking and seek antifungal treatment"
    },
    "Darier Disease": {
        "description": "A rare genetic skin disorder affecting nails.",
        "symptoms": "Red and white nail streaks, nail fragility",
        "causes": "Genetic mutation",
        "precautions": "Dermatologist consultation"
    },
    "Half and Half Nails": {
        "description": "Nails appear half white and half red or brown.",
        "symptoms": "Color separation in nails",
        "causes": "Chronic kidney disease",
        "precautions": "Medical evaluation recommended"
    },
    "Muehrcke's Lines": {
        "description": "Paired white lines across nails.",
        "symptoms": "White transverse nail lines",
        "causes": "Low albumin levels",
        "precautions": "Blood tests and nutrition improvement"
    },
    "Pale Nail": {
        "description": "Unusually pale nail beds.",
        "symptoms": "Pale or white nails",
        "causes": "Anemia or poor circulation",
        "precautions": "Consult a physician"
    },
    "Red Lunula": {
        "description": "Red discoloration of the lunula.",
        "symptoms": "Red half-moon at nail base",
        "causes": "Heart or autoimmune conditions",
        "precautions": "Medical evaluation advised"
    },
    "Splinter Hemorrhage": {
        "description": "Tiny blood clots under the nails.",
        "symptoms": "Thin red or brown lines",
        "causes": "Trauma or systemic disease",
        "precautions": "Monitor and seek medical advice"
    },
    "Terry's Nail": {
        "description": "Mostly white nails with a narrow pink band.",
        "symptoms": "White nails with pink edges",
        "causes": "Liver disease or diabetes",
        "precautions": "Medical consultation required"
    },
    "White Nail": {
        "description": "Complete whitening of the nail plate.",
        "symptoms": "White nails",
        "causes": "Aging or systemic disease",
        "precautions": "Consult a doctor if persistent"
    }
}


# =======================
# Routes
# =======================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    details = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # -------- Image Preprocessing --------
            img = image.load_img(save_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # -------- Prediction --------
            preds = model.predict(img_array)
            predicted_index = np.argmax(preds)
            confidence = round(float(np.max(preds)) * 100, 2)
            prediction = class_names[predicted_index]

            # -------- Disease Details --------
            details = disease_info.get(
                prediction,
                {
                    "description": "No detailed information available.",
                    "symptoms": "N/A",
                    "causes": "N/A",
                    "precautions": "Consult a healthcare professional."
                }
            )

            image_path = f"uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        details=details
    )


# =======================
# Run App
# =======================
if __name__ == "__main__":
    app.run(debug=True)
