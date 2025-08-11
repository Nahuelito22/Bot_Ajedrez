import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def crear_modelo_rnn(vocab_size, embedding_dim, max_sequence_length, embedding_matrix):
    """
    Crea y compila el modelo RNN para el bot de ajedrez.
    """
    model = Sequential()

    # Capa de embedding: Usamos el embedding preentrenado de Word2Vec
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix], # Usamos la matriz de embedding pre-calculada
        input_length=max_sequence_length, # Se necesita para la primera capa
        trainable=True # Permitir que el embedding se ajuste durante el entrenamiento
    ))

    # Capas LSTM y Dropout para regularización
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))

    # Capa densa para la predicción final
    model.add(Dense(vocab_size, activation='softmax'))

    # Compilar el modelo
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', # Mejor para etiquetas enteras
        metrics=['accuracy']
    )

    print("✅ Modelo creado y compilado correctamente.")
    model.summary()
    return model