# train.py (Versión Final con capacidad para REANUDAR)

import os
import argparse
import numpy as np
import tensorflow as tf
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
# --- NUEVO ARGUMENTO PARA REANUDAR ---
parser.add_argument(
    '--resume_from',
    type=str,
    default=None, # Por defecto, no reanudamos desde ningún archivo
    help='Ruta a un checkpoint .keras para reanudar el entrenamiento.'
)

args = parser.parse_args()
DATA_PATH = args.data_path

# --- Constantes y Configuración ---
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
FINAL_MODEL_PATH = os.path.join(DATA_PATH, "chess_lstm_model_final.keras")

EPOCHS = 25
BATCH_SIZE = 512
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
    total_samples = sum(len(seq) - 1 for seq in sequences)
    steps_per_epoch = max(1, total_samples // BATCH_SIZE)
    print(f"Se generarán aproximadamente {total_samples} muestras en total.")
    print(f"Con un tamaño de lote de {BATCH_SIZE}, se realizarán {steps_per_epoch} pasos por época.")

    initial_epoch = 0 # Por defecto, empezamos en la época 0

    # --- Paso 3: CREAR O CARGAR EL MODELO ---
    if args.resume_from:
        print(f"--- Reanudando entrenamiento desde: {args.resume_from} ---")
        model = tf.keras.models.load_model(args.resume_from)
        # Extraemos el número de la época del nombre del archivo para saber dónde continuar
        try:
            filename = os.path.basename(args.resume_from)
            # Asumiendo el formato "modelo_epoch_XX.keras"
            initial_epoch = int(filename.split('_')[2].split('.')[0])
        except (IndexError, ValueError):
            print("ADVERTENCIA: No se pudo determinar la época inicial desde el nombre del archivo.")
            # Si el nombre no sigue el formato, simplemente continuamos desde la siguiente época
            initial_epoch = model.history.epoch[-1] + 1 if model.history and model.history.epoch else 0

        print(f"Se continuará desde la época: {initial_epoch}")
    else:
        print("--- Creando un modelo nuevo ---")
        vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = embedding_matrix.shape[1]
        model = crear_modelo_rnn(vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH, embedding_matrix)
    # -----------------------------------------------

    # --- Paso 4: Configurar Callbacks ---
    print("--- Configurando Callbacks ---")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "modelo_epoch_{epoch:02d}.keras")
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=False)
    early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1)

    # --- Paso 5: Crear el Generador y Entrenar ---
    train_generator = generate_batches_eficiente(sequences, BATCH_SIZE, MAX_SEQUENCE_LENGTH)

    print("\n--- Iniciando Entrenamiento ---")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=initial_epoch, # <--- Le decimos a .fit() dónde empezar
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1
    )

    # --- Paso 6: Guardar el Modelo Final ---
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Entrenamiento completado. Modelo final guardado en '{FINAL_MODEL_PATH}'")