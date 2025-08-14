# main.py (Versión Final con Top-3 Predicciones)

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware

# --- Carga del Modelo y Tokenizador (no cambia) ---
print("Cargando modelo y tokenizador...")
MODEL_PATH = os.path.join("trained_models", "modelo_epoch_06.keras")
TOKENIZER_PATH = os.path.join("trained_models", "tokenizer.pkl")
MAX_SEQUENCE_LENGTH = 50
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("✅ Modelo y tokenizador cargados correctamente.")


# #######################################################################
# ### INICIO DEL CAMBIO ###
# #######################################################################
# --- Definición de los Modelos de Datos ---
class MoveRequest(BaseModel):
    moves: list[str]

class MoveResponse(BaseModel):
    # Ahora la respuesta es una lista de jugadas
    bot_moves: list[str]

# --- Lógica de Predicción ---
def predecir_siguientes_movimientos(sequence_text, top_k=3):
    """
    Toma una secuencia de jugadas y devuelve las K mejores predicciones del modelo.
    """
    sequence_num = tokenizer.texts_to_sequences([sequence_text])
    sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence_num, maxlen=MAX_SEQUENCE_LENGTH, padding='post'
    )
    # [0] para acceder al primer (y único) resultado del lote
    prediction = model.predict(sequence_padded, verbose=0)[0]
    
    # Obtenemos los índices de las K mejores predicciones usando argsort
    # `[-top_k:]` toma los últimos k (los más altos) y `[::-1]` los invierte (de mayor a menor)
    top_indices = np.argsort(prediction)[-top_k:][::-1]
    
    # Convertimos los índices a jugadas en texto
    predicted_moves = [tokenizer.index_word.get(i, "[DESCONOCIDO]") for i in top_indices]
    
    return predicted_moves
# #######################################################################
# ### FIN DEL CAMBIO ###
# #######################################################################


# --- INICIALIZACIÓN DE LA APP Y CORS (no cambia) ---
app = FastAPI()
origins = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "API del bot de ajedrez lista para jugar."}

# --- ENDPOINT MODIFICADO ---
@app.post("/predict_move", response_model=MoveResponse)
def predict_move(request: MoveRequest):
    print(f"Jugadas recibidas: {request.moves}")
    
    # Usamos la nueva función para obtener las 3 mejores jugadas
    jugadas_del_bot = predecir_siguientes_movimientos(request.moves, top_k=3)
    
    print(f"El bot propone estas jugadas (en orden): {jugadas_del_bot}")
    
    # Devolvemos la lista completa de jugadas
    return MoveResponse(bot_moves=jugadas_del_bot)
# -------------------------


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)