# main.py (Versión Final con CORS habilitado)

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
import pickle
import os
# --- 1. NUEVO IMPORT ---
from fastapi.middleware.cors import CORSMiddleware

# ... (El código que carga el modelo y el tokenizador no cambia) ...
print("Cargando modelo y tokenizador...")
MODEL_PATH = os.path.join("trained_models", "modelo_epoch_06.keras")
TOKENIZER_PATH = os.path.join("trained_models", "tokenizer.pkl")
MAX_SEQUENCE_LENGTH = 50
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("✅ Modelo y tokenizador cargados correctamente.")

# --- Definición de los Modelos de Datos (no cambia) ---
class MoveRequest(BaseModel):
    moves: list[str]
class MoveResponse(BaseModel):
    bot_move: str

# ... (La función de predicción no cambia) ...
def predecir_siguiente_movimiento(sequence_text):
    sequence_num = tokenizer.texts_to_sequences([sequence_text])
    sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence_num, maxlen=MAX_SEQUENCE_LENGTH, padding='post'
    )
    prediction = model.predict(sequence_padded, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_move = tokenizer.index_word.get(predicted_index, "[DESCONOCIDO]")
    return predicted_move

# --- INICIALIZACIÓN DE LA APP ---
app = FastAPI()

# --- 2. CONFIGURACIÓN DE CORS ---
# Lista de "orígenes" (países) que tienen permiso para llamar a nuestra API
origins = [
    "http://localhost:8001", # El origen de tu frontend
    "http://127.0.0.1:8001",  # A veces el navegador usa esta dirección
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (GET, POST, etc)
    allow_headers=["*"], # Permite todas las cabeceras
)
# ---------------------------------


# --- 3. ENDPOINTS (no cambian) ---
@app.get("/")
def read_root():
    return {"message": "API del bot de ajedrez lista para jugar."}

@app.post("/predict_move", response_model=MoveResponse)
def predict_move(request: MoveRequest):
    print(f"Jugadas recibidas: {request.moves}")
    jugada_del_bot = predecir_siguiente_movimiento(request.moves)
    print(f"El bot responde: {jugada_del_bot}")
    return MoveResponse(bot_move=jugada_del_bot)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)