import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def cargar_artefactos_entrenamiento(base_path):
    """Carga los artefactos necesarios para el entrenamiento desde archivos.

    Lee el tokenizador, las secuencias de movimientos y la matriz de embedding
    pre-entrenada desde los archivos especificados.

    Args:
        base_path (str): La ruta base donde se encuentran los archivos
                         'tokenizer.pkl', 'sequences.pkl' y 'embedding_matrix.npy'.

    Returns:
        tuple: Una tupla conteniendo:
            - tokenizer (keras.preprocessing.text.Tokenizer): El tokenizador cargado.
            - sequences (list): La lista de secuencias de movimientos.
            - embedding_matrix (np.ndarray): La matriz de embedding.
    """
    print("Cargando artefactos de entrenamiento...")
    with open(f"{base_path}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(f"{base_path}/sequences.pkl", "rb") as f:
        sequences = pickle.load(f)

    embedding_matrix = np.load(f"{base_path}/embedding_matrix.npy")

    print("✅ Artefactos cargados.")
    return tokenizer, sequences, embedding_matrix

def crear_pares_entrenamiento(sequences):
    """Transforma una lista de secuencias en pares de entrada (X) y salida (y).

    Para cada secuencia en la lista, genera múltiples ejemplos de entrenamiento.
    Por ejemplo, la secuencia [a, b, c] generará los pares:
    - X=[a], y=b
    - X=[a, b], y=c

    Este es un paso de pre-procesamiento que se realiza una sola vez para
    acelerar el entrenamiento en cada época.

    Args:
        sequences (list): Una lista de listas, donde cada sublista es una secuencia
                          de tokens (movimientos).

    Returns:
        tuple: Una tupla conteniendo:
            - X (list): Una lista de secuencias de entrada.
            - y (list): Una lista de etiquetas de salida correspondientes.
    """
    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])
    return X, y

def generate_batches(X, y, batch_size, max_sequence_length):
    """Generador que crea lotes de datos para el entrenamiento de forma eficiente.

    Toma los pares de (X, y) pre-procesados, los mezcla y los divide en lotes
    del tamaño especificado. Aplica padding a las secuencias de entrada (X) de
    cada lote justo antes de entregarlo.

    Args:
        X (list): La lista completa de secuencias de entrada.
        y (list): La lista completa de etiquetas de salida.
        batch_size (int): El número de muestras por lote.
        max_sequence_length (int): La longitud máxima a la que se deben rellenar
                                   (pad) las secuencias de entrada.

    Yields:
        tuple: Una tupla conteniendo (X_batch, y_batch):
            - X_batch (np.ndarray): Un lote de secuencias de entrada, con padding
                                    y con forma (batch_size, max_sequence_length).
            - y_batch (np.ndarray): Un lote de etiquetas de salida, con forma
                                    (batch_size,).
    """
    num_samples = len(X)
    # Mezclar los datos al inicio de cada época para mejorar el aprendizaje
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        # Extraer el lote usando los índices mezclados
        X_batch = [X[i] for i in batch_indices]
        y_batch = [y[i] for i in batch_indices]

        # Aplicar padding solo al lote actual
        X_padded = pad_sequences(X_batch, maxlen=max_sequence_length, padding='post')

        yield np.array(X_padded), np.array(y_batch)
