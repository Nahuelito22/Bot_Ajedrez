import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def crear_modelo_rnn(vocab_size, embedding_dim, max_sequence_length, embedding_matrix):
    """
    Crea y compila un modelo RNN mejorado para el bot de ajedrez utilizando la API Funcional de Keras.

    Esta versión incluye Batch Normalization para una mejor estabilidad y convergencia del entrenamiento.
    """
    # --- 1. Definición de la Entrada del Modelo ---
    # Usar la API Funcional empieza con una capa de Input, que hace la forma de los datos de entrada explícita.
    input_layer = Input(shape=(max_sequence_length,), name='movimientos_input')

    # --- 2. Capa de Embedding ---
    # La capa de Embedding convierte las secuencias de enteros (movimientos) en vectores densos.
    # Usamos la matriz de embedding pre-calculada, pero permitimos que se ajuste (fine-tuning).
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_sequence_length,
        trainable=True,
        name='embedding'
    )(input_layer)

    # --- 3. Capas LSTM con Regularización ---
    # Primera capa LSTM. `return_sequences=True` es necesario para pasar la secuencia completa a la siguiente capa LSTM.
    lstm1 = LSTM(128, return_sequences=True, name='lstm_1')(embedding_layer)
    # Batch Normalization ayuda a estabilizar y acelerar el entrenamiento. Se aplica después de la capa recurrente.
    bn1 = BatchNormalization(name='batch_norm_1')(lstm1)
    # Dropout es una técnica de regularización para prevenir el sobreajuste.
    drop1 = Dropout(0.2, name='dropout_1')(bn1)

    # Segunda capa LSTM. Ya no necesita `return_sequences=True` porque la siguiente capa es Densa.
    lstm2 = LSTM(128, name='lstm_2')(drop1)
    bn2 = BatchNormalization(name='batch_norm_2')(lstm2)
    drop2 = Dropout(0.2, name='dropout_2')(bn2)

    # --- 4. Capa de Salida ---
    # La capa Densa final predice el siguiente movimiento sobre todo el vocabulario.
    # 'softmax' convierte los logits en una distribución de probabilidad.
    output_layer = Dense(vocab_size, activation='softmax', name='output_prediccion')(drop2)

    # --- 5. Creación y Compilación del Modelo ---
    # Se define el modelo especificando las capas de entrada y salida.
    model = Model(inputs=input_layer, outputs=output_layer, name='BotAjedrezRNN')

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Ideal para clasificación con etiquetas enteras.
        metrics=['accuracy']
    )

    print("✅ Modelo creado y compilado con la API Funcional.")
    model.summary()
    return model
