# Bot de Ajedrez con Redes Neuronales LSTM ♟️

Un bot de ajedrez diseñado para jugar con un estilo "humano", aprendiendo de una base de datos de 1 millón de partidas de jugadores de alto nivel (2000+ ELO) de la plataforma Lichess.

A diferencia de los motores de ajedrez tradicionales como Stockfish que se basan en la fuerza bruta y el cálculo de árboles de búsqueda, este proyecto utiliza una **Red Neuronal Recurrente (LSTM)** para predecir el siguiente movimiento más probable, imitando la intuición y el conocimiento posicional de jugadores experimentados.

## 🌟 Características

* **Modelo Predictivo LSTM:** El núcleo del bot es una red neuronal profunda que analiza secuencias de movimientos para predecir la siguiente jugada.
* **Entrenamiento de Alta Calidad:** Entrenado sobre un corpus masivo y filtrado de partidas de grandes maestros y jugadores titulados.
* **API para Jugabilidad:** El modelo se expondrá a través de una API RESTful para facilitar la integración con diferentes interfaces.
* **Interfaz Web (En Desarrollo):** El objetivo final es crear una interfaz web donde se pueda jugar directamente contra el bot.



## 🏗️ Arquitectura del Proyecto

El proyecto está diseñado con una arquitectura moderna de tres componentes principales:

1.  **Modelo de IA (`model.py`):** La red neuronal construida con TensorFlow/Keras que constituye el cerebro del bot.
2.  **Backend (`main.py`):** Un servidor web construido con **FastAPI** que carga el modelo entrenado y expone un endpoint (ej. `/move`) para recibir una posición y devolver la jugada del bot.
3.  **Frontend (`index.html`, etc.):** Una interfaz de usuario interactiva en el navegador, desarrollada con **HTML, CSS y JavaScript**, utilizando librerías como `Chess.js` y `Chessboard.js` para la lógica y visualización del tablero.

## 🛠️ Stack Tecnológico

* **Backend:** Python 3.11, FastAPI
* **Machine Learning:** TensorFlow, Keras, NumPy, Scikit-learn
* **Manejo de Lógica de Ajedrez:** `python-chess`
* **Entorno:** JupyterLab, Google Colab (para entrenamiento con GPU)
* **Frontend (planeado):** JavaScript, Chess.js, Chessboard.js

## 🚀 Instalación y Configuración Local

Para configurar el entorno y ejecutar el proyecto localmente, sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://www.youtube.com/watch?v=3fn7ApOWE1k](https://www.youtube.com/watch?v=3fn7ApOWE1k)
    cd Bot_Ajedrez
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    # Se recomienda usar Python 3.11
    py -3.11 -m venv venv
    venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Uso

### Entrenamiento

El entrenamiento es un proceso intensivo que se realiza en un entorno con GPU (como Google Colab o Kaggle). El script `train.py` está preparado para reanudar el entrenamiento desde checkpoints.

```bash
# Para iniciar un nuevo entrenamiento
python train.py --data_path "ruta/a/tus/datos"

# Para reanudar desde un checkpoint
python train.py --data_path "ruta/a/tus/datos" --resume_from "ruta/a/checkpoints/modelo_epoch_XX.keras"