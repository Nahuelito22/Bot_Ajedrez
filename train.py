# 1. Imports de librerías y módulos
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from model import crear_modelo_rnn
from data_loader import (
                         cargar_artefactos_entrenamiento,
                         crear_pares_entrenamiento,
                         generate_batches)

# 2. Constantes y Configuración
BASE_PATH = "."  # Asumimos que los datos están en el directorio actual
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
LOG_FILE = os.path.join(BASE_PATH, "historial_entrenamiento.csv")
FINAL_MODEL_PATH = os.path.join(BASE_PATH, "chess_lstm_model_final.keras")

EPOCHS = 25  # Aumentamos las épocas, EarlyStopping se encargará de parar si es necesario
BATCH_SIZE = 128
MAX_SEQUENCE_LENGTH = 50
VALIDATION_SPLIT = 0.1 # Usaremos un 10% de los datos para validación

# Crear directorio para checkpoints si no existe
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# 3. Lógica Principal de Entrenamiento
if __name__ == "__main__":
    # --- Paso 1: Cargar y Preparar los Datos ---
    print("--- Cargando y preparando datos ---")
    tokenizer, sequences, embedding_matrix = cargar_artefactos_entrenamiento(BASE_PATH)
    
    # Crear todos los pares (X, y) una sola vez
    X, y = crear_pares_entrenamiento(sequences)

    # Dividir en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Datos de entrenamiento: {len(X_train)} muestras")
    print(f"Datos de validación: {len(y_val)} muestras")

    # --- Paso 2: Crear el Modelo ---
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = embedding_matrix.shape[1]
    model = crear_modelo_rnn(vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH, embedding_matrix)

    # --- Paso 3: Configurar Callbacks ---
    print("--- Configurando Callbacks ---")
    # Guarda el mejor modelo basado en la pérdida de validación
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "mejor_modelo_epoch_{epoch:02d}-val_loss_{val_loss:.2f}.keras")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Registra el historial de entrenamiento en un archivo CSV
    csv_logger = CSVLogger(LOG_FILE)

    # Detiene el entrenamiento si no hay mejora
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3, # Número de épocas sin mejora antes de parar
        verbose=1,
        mode='min',
        restore_best_weights=True # Restaura los pesos del mejor modelo al final
    )

    # --- Paso 4: Crear Generadores de Datos ---
    train_generator = generate_batches(X_train, y_train, BATCH_SIZE, MAX_SEQUENCE_LENGTH)
    val_generator = generate_batches(X_val, y_val, BATCH_SIZE, MAX_SEQUENCE_LENGTH)

    # Calcular los pasos por época
    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE
    if steps_per_epoch == 0: steps_per_epoch = 1
    if validation_steps == 0: validation_steps = 1

    # --- Paso 5: Iniciar el Entrenamiento ---
    print("\n--- Iniciando Entrenamiento ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[model_checkpoint, csv_logger, early_stopping],
        verbose=1
    )

    # --- Paso 6: Guardar el Modelo Final ---
    # El callback EarlyStopping con `restore_best_weights=True` ya ha restaurado
    # los mejores pesos en el modelo, por lo que guardamos el estado óptimo.
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Entrenamiento completado. Modelo final guardado en '{FINAL_MODEL_PATH}'")
