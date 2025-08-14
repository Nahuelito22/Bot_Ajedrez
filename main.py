# main.py (Versión con IA Real)

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
import pickle
import os

# --- 1. Carga del Modelo y el Tokenizador ---
# Estas líneas se ejecutan una sola vez, cuando la API se inicia.

print("Cargando modelo y tokenizador...")
MODEL_PATH = os.path.join("trained_models", "modelo_epoch_06.keras")
TOKENIZER_PATH = os.path.join("trained_models", "tokenizer.pkl")
MAX_SEQUENCE_LENGTH = 50 # La misma que usaste en el entrenamiento

# Cargamos el modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)

# Cargamos el tokenizador
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

print("✅ Modelo y tokenizador cargados correctamente.")
# ---------------------------------------------


# --- 2. Definición de los Modelos de Datos (Request/Response) ---
class MoveRequest(BaseModel):
    moves: list[str]

class MoveResponse(BaseModel):
    bot_move: str

# -----------------------------------------

# --- 3. Lógica de Predicción ---
def predecir_siguiente_movimiento(sequence_text):
    """
    Toma una secuencia de jugadas en texto, las procesa y devuelve
    la predicción del modelo.
    """
    # Convertir la secuencia de texto a una secuencia de números
    sequence_num = tokenizer.texts_to_sequences([sequence_text])
    
    # Hacer padding para que tenga la longitud correcta
    sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence_num, maxlen=MAX_SEQUENCE_LENGTH, padding='post'
    )
    
    # Hacer la predicción
    prediction = model.predict(sequence_padded, verbose=0)
    
    # Obtener el índice de la jugada con la probabilidad más alta
    predicted_index = np.argmax(prediction)
    
    # Convertir el índice de vuelta a texto
    predicted_move = tokenizer.index_word.get(predicted_index, "[DESCONOCIDO]")
    
    return predicted_move
# -----------------------------------------

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API del bot de ajedrez lista para jugar."}


@app.post("/predict_move", response_model=MoveResponse)
def predict_move(request: MoveRequest):
    """
    Recibe una secuencia de jugadas y devuelve la siguiente jugada del bot.
    """
    print(f"Jugadas recibidas: {request.moves}")
    
    # Usamos nuestra función de predicción
    jugada_del_bot = predecir_siguiente_movimiento(request.moves)
    
    print(f"El bot responde: {jugada_del_bot}")
    
    return MoveResponse(bot_move=jugada_del_bot)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)