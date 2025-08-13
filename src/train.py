# train.py (Versión Final para Kaggle, separando lectura y escritura)

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.model import crear_modelo_rnn
from data_loader import cargar_artefactos_entrenamiento, generate_batches_eficiente

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Entrenar el modelo de ajedrez LSTM.")
parser.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='Ruta a la carpeta de DATOS de solo lectura (/kaggle/input/...).'
)
parser.add_argument(
    '--resume_from',
    type=str,
    default=None,
    help='Ruta a un checkpoint .keras para reanudar el entrenamiento.'
)
args = parser.parse_args()
DATA_PATH = args.data_path # <-- Esta es la ruta para LEER datos

# #######################################################################
# ### INICIO DEL CAMBIO ###
# #######################################################################
# --- RUTAS DE SALIDA ---
# Todos los archivos que se escriban irán a /kaggle/working/
OUTPUT_DIR = "/kaggle/working/"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "chess_lstm_model_final.keras")

# Crear directorio para checkpoints en la carpeta de SALIDA
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
# #######################################################################
# ### FIN DEL CAMBIO ###
# #######################################################################

# --- Constantes ---
EPOCHS = 25
BATCH_SIZE = 512
MAX_SEQUENCE_LENGTH = 50

# --- Lógica Principal de Entrenamiento ---
if __name__ == "__main__":
    # --- Paso 1: Cargar los Datos (desde DATA_PATH) ---
    print(f"--- Cargando datos desde: {DATA_PATH} ---")
    tokenizer, sequences = cargar_artefactos_entrenamiento(DATA_PATH)
    embedding_matrix = np.load(os.path.join(DATA_PATH, "embedding_matrix.npy"))

    # --- Paso 2: Calcular Parámetros ---
    total_samples = sum(len(seq) - 1 for seq in sequences)
    steps_per_epoch = max(1, total_samples // BATCH_SIZE)
    print(f"Se generarán aproximadamente {total_samples} muestras en total.")
    print(f"Con un tamaño de lote de {BATCH_SIZE}, se realizarán {steps_per_epoch} pasos por época.")

    initial_epoch = 0

    # --- Paso 3: Crear o Cargar Modelo (Carga desde la ruta de entrada) ---
    if args.resume_from:
        print(f"--- Reanudando entrenamiento desde: {args.resume_from} ---")
        model = tf.keras.models.load_model(args.resume_from)
        try:
            filename = os.path.basename(args.resume_from)
            initial_epoch = int(filename.split('_')[2].split('.')[0])
        except (IndexError, ValueError):
            print("ADVERTENCIA: No se pudo determinar la época inicial del nombre del archivo.")
            initial_epoch = 0
        print(f"Se continuará a partir de la época: {initial_epoch}")
    else:
        print("--- Creando un modelo nuevo ---")
        vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = embedding_matrix.shape[1]
        model = crear_modelo_rnn(vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH, embedding_matrix)

    # --- Paso 4: Configurar Callbacks (Guardan en la ruta de SALIDA) ---
    print(f"--- Configurando Callbacks para guardar en: {CHECKPOINT_DIR} ---")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "modelo_epoch_{epoch:02d}.keras")
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=False, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True)

    # --- Paso 5: Crear el Generador y Entrenar ---
    train_generator = generate_batches_eficiente(sequences, BATCH_SIZE, MAX_SEQUENCE_LENGTH)

    print("\n--- Iniciando Entrenamiento ---")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1
    )

    # --- Paso 6: Guardar el Modelo Final (en la ruta de SALIDA) ---
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Entrenamiento completado. Modelo final guardado en '{FINAL_MODEL_PATH}'")