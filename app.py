from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io, base64, os
import gdown

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "https://oncho-scan.vercel.app", 
    "*" 
])

IMG_SIZE   = (224, 224)
CLASS_NAMES = ['benign', 'malignant', 'normal']
MODEL_PATH = 'best_model.keras'
FILE_ID    = '1akIlYQLk7AVunBhjGyx3fV9v54MSxHb3'

if not os.path.exists(MODEL_PATH):
    print("⏳ Downloading model from Drive...")
    gdown.download(f'https://drive.google.com/uc?id={FILE_ID}', MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

print("⏳ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")
print("   Input shape: ", model.input_shape)
print("   Output shape:", model.output_shape)

@app.route('/')
def home():
    return jsonify({"message": "Breast Cancer Detection API Running"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data       = request.json['image']
        img_bytes  = base64.b64decode(data.split(',')[1])
        img        = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(IMG_SIZE)
        img_array  = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        print("Input Shape:", img_array.shape)

        predictions   = model.predict(img_array)
        max_index     = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[max_index]
        confidence    = float(predictions[0][max_index])

        print("Predicted Class:", predicted_class, "| Confidence:", confidence)

        return jsonify({
            "class":      predicted_class,
            "confidence": confidence,
            "all_scores": {
                "Benign":    float(predictions[0][0]),
                "Malignant": float(predictions[0][1]),
                "Normal":    float(predictions[0][2])
            }
        })
    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)