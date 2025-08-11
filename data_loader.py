import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def cargar_artefactos_entrenamiento(base_path):
    """
    Carga el tokenizador, las secuencias y la matriz de embedding desde Google Drive.
    """
    print("Cargando artefactos de entrenamiento...")
    # Cargar el tokenizador
    with open(f"{base_path}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Cargar las secuencias
    with open(f"{base_path}/sequences.pkl", "rb") as f:
        sequences = pickle.load(f)

    # Cargar la matriz de embedding
    embedding_matrix = np.load(f"{base_path}/embedding_matrix.npy")

    print("✅ Artefactos cargados.")
    return tokenizer, sequences, embedding_matrix

def generate_batches(sequences, batch_size, max_sequence_length):
    """
    Generador que crea lotes de secuencias y etiquetas para el entrenamiento.
    """
    input_sequences = []
    output_labels = []

    for sequence in sequences:
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            output_labels.append(sequence[i])

            if len(input_sequences) >= batch_size:
                input_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
                yield np.array(input_padded), np.array(output_labels)
                input_sequences, output_labels = [], []

    if input_sequences:
        input_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
        yield np.array(input_padded), np.array(output_labels)