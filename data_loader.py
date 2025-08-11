# data_loader.py (Versión Final Eficiente)

import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

def cargar_artefactos_entrenamiento(base_path):
    """
    Carga los artefactos necesarios para el entrenamiento desde archivos.

    Lee el tokenizador y las secuencias de movimientos. La matriz de embedding
    se cargará por separado en el script de entrenamiento donde se necesita.

    Args:
        base_path (str): La ruta base donde se encuentran 'tokenizer.pkl' y 'sequences.pkl'.

    Returns:
        tuple: Una tupla conteniendo (tokenizer, sequences).
    """
    print("Cargando artefactos de entrenamiento...")
    with open(f"{base_path}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(f"{base_path}/sequences.pkl", "rb") as f:
        sequences = pickle.load(f)

    print("✅ Artefactos cargados.")
    return tokenizer, sequences

def generate_batches_eficiente(sequences, batch_size, max_sequence_length):
    """
    Generador que crea lotes de datos para el entrenamiento de forma eficiente,
    procesando las secuencias "al vuelo" para no consumir toda la RAM.

    Args:
        sequences (list): La lista completa de secuencias de movimientos.
        batch_size (int): El número de muestras por lote.
        max_sequence_length (int): La longitud máxima para el padding.

    Yields:
        tuple: Una tupla (X_batch, y_batch) lista para el entrenamiento.
    """
    # Usamos un bucle infinito para que el generador pueda usarse en múltiples épocas
    while True:
        # Mezclar las secuencias al inicio de cada pasada completa sobre los datos
        random.shuffle(sequences)

        X_batch, y_batch = [], []

        # Iteramos sobre cada partida (secuencia)
        for seq in sequences:
            # Creamos los pares de (entrada, salida) para esa partida
            for i in range(1, len(seq)):
                X_batch.append(seq[:i])
                y_batch.append(seq[i])

                # Cuando juntamos suficientes muestras para un lote, lo procesamos y entregamos
                if len(X_batch) == batch_size:
                    X_padded = pad_sequences(X_batch, maxlen=max_sequence_length, padding='post')
                    yield np.array(X_padded), np.array(y_batch)
                    
                    # Limpiamos las listas para el siguiente lote
                    X_batch, y_batch = [], []