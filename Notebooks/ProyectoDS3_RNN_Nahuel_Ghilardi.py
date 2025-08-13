import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

dataset_path = '/content/drive/MyDrive/CoderHouse/Modulo 3/movimientos_tokenizados.csv'
df = pd.read_csv(dataset_path)

df.info()

from gensim.models import Word2Vec
import ast

# Tokenizar y preparar las jugadas como listas de tokens
movements = df['movetext'].apply(ast.literal_eval).tolist()

# Configurar y entrenar el modelo Word2Vec
model = Word2Vec(sentences=movements, vector_size=300, window=5, min_count=1, sg=1)

# Guardar el modelo entrenado
model.save("chess_300d_word2vec.model")

print(model.wv['d4'])  # Vector para la jugada 'e4'

# Jugadas más similares a 'e4'
similar_moves = model.wv.most_similar('e4', topn=5)
print("Jugadas más similares a 'e4':", similar_moves)

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Palabras (movimientos) a analizar
words = ['e4', 'd4', 'c4', 'Nf3', 'Nc3', 'a3', 'h3']
vectors = np.array([model.wv[word] for word in words])  # Convertir a numpy array

# Reducir a 2 dimensiones usando t-SNE
tsne = TSNE(n_components=2, perplexity=5, max_iter=300, random_state=42)  # n_iter cambiado a max_iter
reduced_vectors = tsne.fit_transform(vectors)

# Graficar
plt.figure(figsize=(10, 6))
for word, (x, y) in zip(words, reduced_vectors):
    plt.scatter(x, y, label=word)
    plt.text(x + 0.05, y + 0.05, word, fontsize=12)

plt.title("Visualización del espacio de embedding con t-SNE")
plt.legend()
plt.show()


from gensim.models import Word2Vec

# Ruta del archivo en Google Drive
model_path = "/content/drive/MyDrive/CoderHouse/Modulo 3/chess_300d_word2vec.model"

# Cargar el modelo desde la ruta
model = Word2Vec.load(model_path)

# Verificar que está cargado correctamente
print(model.wv['e4'])

# !pip install tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

# Convertir movements a una lista de cadenas de texto
movements_text = [' '.join(move) for move in movements]

# Inicializar el tokenizador
tokenizer = Tokenizer(filters='', lower=False)
tokenizer.fit_on_texts(movements_text)  # Usar movements_text en lugar de df['movetext']

# Convertir los movimientos a secuencias de índices
sequences = tokenizer.texts_to_sequences(movements_text)

# Guardar el tamaño del vocabulario
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulario: {vocab_size} palabras únicas")


# Vocabulario del tokenizador
vocab_tokenizer = set(tokenizer.word_index.keys())

# Cargar el modelo preentrenado
embedding_model = Word2Vec.load('/content/drive/MyDrive/CoderHouse/Modulo 3/chess_300d_word2vec.model')
print(f"Vocabulario cargado: {len(embedding_model.wv)} palabras únicas.")

# Vocabulario de Word2Vec
vocab_word2vec = set(embedding_model.wv.key_to_index.keys())

# Palabras en el tokenizador pero no en Word2Vec
palabras_faltantes = vocab_tokenizer - vocab_word2vec
print(f"Palabras en el tokenizador pero no en Word2Vec: {len(palabras_faltantes)}")
print(f"Ejemplo de palabras faltantes: {list(palabras_faltantes)[:10]}")

def generate_batches(sequences, batch_size=10000, max_sequence_length=None):
    """
    Generador que crea lotes de secuencias y etiquetas sin cargar todo en memoria.
    Añadimos padding para asegurar que todas las secuencias tienen la misma longitud.
    """
    input_sequences = []
    output_labels = []

    # Determinamos la longitud máxima si no se pasa como argumento
    if max_sequence_length is None:
        max_sequence_length = max(len(sequence) for sequence in sequences)

    for sequence in sequences:
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            output_labels.append(sequence[i])

            # Si hemos alcanzado el tamaño del batch, lo devolvemos
            if len(input_sequences) >= batch_size:
                # Hacer padding a las secuencias de entrada para que tengan la misma longitud
                input_sequences_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
                yield np.array(input_sequences_padded), np.array(output_labels)
                input_sequences = []  # Limpiar para el siguiente lote
                output_labels = []

    # Devolver cualquier secuencia restante que no haya alcanzado el tamaño del batch
    if input_sequences:
        input_sequences_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
        yield np.array(input_sequences_padded), np.array(output_labels)

# Usar el generador
batch_size = 10000  # Ajusta el tamaño del lote según tus necesidades
max_sequence_length = 50  # Aquí defines la longitud máxima de las secuencias (ajústalo según sea necesario)

for X_batch, y_batch in generate_batches(sequences, batch_size, max_sequence_length):
    # Aquí puedes usar X_batch y y_batch para entrenar tu modelo
    print(f"Lote de secuencias X: {X_batch.shape}, Lote de etiquetas y: {y_batch.shape}")


from gensim.models import Word2Vec
import numpy as np

# Cargar el modelo preentrenado
embedding_model = Word2Vec.load('/content/drive/MyDrive/CoderHouse/Modulo 3/chess_300d_word2vec.model')
print(f"Vocabulario cargado: {len(embedding_model.wv)} palabras únicas.")

# Inicializamos la matriz de embeddings con ceros
embedding_dim = 300  # Dimensión del embedding (300 en tu caso)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Llenamos la matriz con los vectores de Word2Vec
for word, index in tokenizer.word_index.items():
    if word in embedding_model.wv:
        embedding_matrix[index] = embedding_model.wv[word]  # Usar el vector de Word2Vec
    else:
        # Asignar un vector aleatorio para palabras desconocidas
        embedding_matrix[index] = np.random.rand(embedding_dim)

# Verifica que la matriz tiene la forma correcta
print(f"Matriz de embeddings generada de tamaño: {embedding_matrix.shape}")


# Verificar el vector de padding (índice 0)
print("Vector de padding (índice 0):", embedding_matrix[0])

# Verificar el vector de una palabra conocida (por ejemplo, 'e4')
word = 'e4'
index = tokenizer.word_index[word]
print(f"Vector para '{word}' (índice {index}):", embedding_matrix[index])

from gensim.models import Word2Vec

# Guardar el modelo de Word2Vec en Google Drive
embedding_model.save("/content/drive/MyDrive/CoderHouse/Modulo 3/chess_300d_word2vec.model")

import numpy as np

# Guardar la matriz de embedding en Google Drive
np.save("/content/drive/MyDrive/CoderHouse/Modulo 3/embedding_matrix.npy", embedding_matrix)

import pickle

# Guardar las secuencias tokenizadas en Google Drive
with open("/content/drive/MyDrive/CoderHouse/Modulo 3/sequences.pkl", "wb") as f:
    pickle.dump(sequences, f)

import pickle

# Guardar el tokenizador en Google Drive
with open("/content/drive/MyDrive/CoderHouse/Modulo 3/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

from google.colab import drive
drive.mount('/content/drive')

from gensim.models import Word2Vec

# Cargar el modelo de Word2Vec desde Google Drive
embedding_model = Word2Vec.load("/content/drive/MyDrive/CoderHouse/Modulo 3/chess_300d_word2vec.model")

import numpy as np

# Cargar la matriz de embedding desde Google Drive
embedding_matrix = np.load("/content/drive/MyDrive/CoderHouse/Modulo 3/embedding_matrix.npy")

import pickle

# Cargar las secuencias tokenizadas desde Google Drive
with open("/content/drive/MyDrive/CoderHouse/Modulo 3/sequences.pkl", "rb") as f:
    sequences = pickle.load(f)

import pickle

# Cargar el tokenizador desde Google Drive
with open("/content/drive/MyDrive/CoderHouse/Modulo 3/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def generate_batches(sequences, batch_size=10000, max_sequence_length=None):
    input_sequences = []
    output_labels = []

    if max_sequence_length is None:
        max_sequence_length = max(len(sequence) for sequence in sequences)

    for sequence in sequences:
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            output_labels.append(sequence[i])

            if len(input_sequences) >= batch_size:
                input_sequences_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
                yield np.array(input_sequences_padded), np.array(output_labels)
                input_sequences = []
                output_labels = []

    if input_sequences:
        input_sequences_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
        yield np.array(input_sequences_padded), np.array(output_labels)

vocab_size = len(tokenizer.word_index) + 1  # Tamaño del vocabulario
embedding_dim = 300  # Dimensión del embedding
max_sequence_length = 50  # Longitud máxima de las secuencias

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Crear el modelo
model = Sequential()

# Capa de embedding: Usamos el embedding preentrenado de Word2Vec
model.add(Embedding(
    input_dim=vocab_size,          # Tamaño del vocabulario
    output_dim=embedding_dim,      # Dimensiones del embedding
    weights=[embedding_matrix],    # Usamos la matriz de embedding que preparamos antes
    trainable=True                 # Permitir que el embedding se ajuste durante el entrenamiento
))

# Primera capa LSTM con regularización L2
model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))

# Segunda capa LSTM con regularización L2
model.add(LSTM(128, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))

# Capa densa para la predicción final
model.add(Dense(vocab_size, activation='softmax'))

# Compilar el modelo con una tasa de aprendizaje ajustada
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Resumen del modelo
model.summary()

# Compilar el modelo
model.compile(
    optimizer='adam',                 # Puedes ajustar el optimizador si lo deseas
    loss='sparse_categorical_crossentropy',  # Pérdida para clasificación multiclase
    metrics=['accuracy']             # Métrica para evaluar
)

print("Modelo compilado correctamente.")


from keras.callbacks import EarlyStopping

# Parámetros de entrenamiento
epochs = 5  # Número de épocas
batch_size = 10000  # Tamaño del lote para generar datos
mini_batch_size = 128  # Tamaño del mini-batch para entrenar dentro de cada lote
max_sequence_length = 50  # Longitud máxima de las secuencias

# Callback para detener el entrenamiento si no mejora
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Entrenamiento
for epoch in range(epochs):
    print(f"\nÉpoca {epoch + 1}/{epochs}")

    # Contador para mostrar el progreso dentro de la época
    batch_counter = 0

    for X_batch, y_batch in generate_batches(sequences, batch_size=batch_size, max_sequence_length=max_sequence_length):
        # Entrenar con el lote actual usando train_on_batch
        loss, accuracy = model.train_on_batch(X_batch, y_batch)

        # Mostrar el progreso
        batch_counter += 1
        print(f"Lote {batch_counter}: Pérdida = {loss:.4f}, Precisión = {accuracy:.4f}")

    print(f"Época {epoch + 1} completada.")

    # Verificar si se debe detener el entrenamiento (early stopping)
    if early_stopping.model is not None and early_stopping.stopped_epoch > 0:
        print("Entrenamiento detenido por early stopping.")
        break

# Finalizar y guardar el modelo
model.save("/content/drive/My Drive/chess_lstm_model.h5")
print("Modelo guardado.")

