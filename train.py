# 1. Imports de nuestros módulos y librerías
from model import crear_modelo_rnn
from data_loader import cargar_artefactos_entrenamiento, generate_batches

# 2. Constantes y Configuración
DRIVE_BASE_PATH = "/content/drive/MyDrive/CoderHouse/Modulo 3"
EPOCHS = 5
BATCH_SIZE = 10000
MAX_SEQUENCE_LENGTH = 50

# 3. Lógica Principal de Entrenamiento
if __name__ == "__main__":
    # Cargar los datos necesarios para entrenar
    tokenizer, sequences, embedding_matrix = cargar_artefactos_entrenamiento(DRIVE_BASE_PATH)

    # Calcular variables necesarias para el modelo
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = embedding_matrix.shape[1]

    # Crear la instancia del modelo
    model = crear_modelo_rnn(vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH, embedding_matrix)

    # Iniciar el entrenamiento
    print("\n--- Iniciando Entrenamiento ---")
    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{EPOCHS}")
        batch_counter = 0
        data_generator = generate_batches(sequences, batch_size=BATCH_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH)

        for X_batch, y_batch in data_generator:
            loss, accuracy = model.train_on_batch(X_batch, y_batch)
            batch_counter += 1
            print(f"  Lote {batch_counter}: Pérdida = {loss:.4f}, Precisión = {accuracy:.4f}")

    # Guardar el modelo final
    model.save(f"{DRIVE_BASE_PATH}/chess_lstm_model_final.keras")
    print("\n✅ Entrenamiento completado. Modelo guardado en 'chess_lstm_model_final.keras'")