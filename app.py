# app.py
import os
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

MODEL_URL = "https://github.com/Seband54/Lie_Truth/releases/download/v1.0/Lie_Truth.keras"
MODEL_PATH = "Lie_Truth.keras"

# Descargar modelo si no existe localmente
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Modelo descargado.")

# Cargar modelo
download_model()
model = load_model(MODEL_PATH)

# Preprocesamiento (64x64 RGB)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0  # Normalización [0,1]
    return np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes_data = await file.read()
    img_batch = preprocess_image(bytes_data)
    prediction = model.predict(img_batch)[0]

    # Asumiendo 2 clases: [prob_verdad, prob_mentira]
    resultado = "Verdad" if prediction[0] > prediction[1] else "Mentira"
    return JSONResponse(content={"resultado": resultado, "predicciones": prediction.tolist()})
