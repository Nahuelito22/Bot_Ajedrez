# train.py (Versión Final para Generador Eficiente)

import os
import argparse
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model import crear_modelo_rnn
from data_loader import cargar_artefactos_entrenamiento, generate_batches_eficiente

# --- Argumentos y Rutas ---
parser = argparse.ArgumentParser(description="Entrenar el modelo de ajedrez LSTM.")
parser.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='Ruta a la carpeta que contiene los datos y donde se guardarán los resultados.'
)
args = parser.parse_args()
DATA_PATH = args.data_path

# --- Constantes y Configuración ---
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
FINAL_MODEL_PATH = os.path.join(DATA_PATH, "chess_lstm_model_final.keras")

EPOCHS = 25
BATCH_SIZE = 512 # Un batch size más pequeño funciona bien con generadores
MAX_SEQUENCE_LENGTH = 50

# Crear directorio para checkpoints si no existe
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# --- Lógica Principal de Entrenamiento ---
if __name__ == "__main__":
    # --- Paso 1: Cargar los Datos ---
    print(f"--- Cargando datos desde: {DATA_PATH} ---")
    tokenizer, sequences = cargar_artefactos_entrenamiento(DATA_PATH)
    embedding_matrix = np.load(os.path.join(DATA_PATH, "embedding_matrix.npy"))

    # --- Paso 2: Calcular Parámetros ---
    # Calculamos el número total de muestras que el generador producirá en una época
    total_samples = sum(len(seq) - 1 for seq in sequences)
    steps_per_epoch = total_samples // BATCH_SIZE
    if steps_per_epoch == 0:
        steps_per_epoch = 1 # Asegurarse de que haya al menos un paso
    
    print(f"Se generarán aproximadamente {total_samples} muestras en total.")
    print(f"Con un tamaño de lote de {BATCH_SIZE}, se realizarán {steps_per_epoch} pasos por época.")

    # --- Paso 3: Crear el Modelo ---
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = embedding_matrix.shape[1]
    model = crear_modelo_rnn(vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH, embedding_matrix)

    # --- Paso 4: Configurar Callbacks ---
    print("--- Configurando Callbacks ---")
    # Guarda un checkpoint después de cada época
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "modelo_epoch_{epoch:02d}.keras")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=False, # Guardamos cada época por si queremos reanudar
        verbose=1
    )

    # Detiene el entrenamiento si la pérdida no mejora
    early_stopping = EarlyStopping(
        monitor='loss', # Monitoreamos la pérdida de entrenamiento
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    # --- Paso 5: Crear el Generador y Entrenar ---
    train_generator = generate_batches_eficiente(sequences, BATCH_SIZE, MAX_SEQUENCE_LENGTH)

    print("\n--- Iniciando Entrenamiento ---")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1
    )

    # --- Paso 6: Guardar el Modelo Final ---
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Entrenamiento completado. Modelo final guardado en '{FINAL_MODEL_PATH}'")