# Bot de Ajedrez con Redes Neuronales LSTM ♟️

Un bot de ajedrez diseñado para jugar con un estilo "humano", aprendiendo de una base de datos de 1 millón de partidas de jugadores de alto nivel (2000+ ELO) de la plataforma Lichess.

A diferencia de los motores de ajedrez tradicionales como Stockfish que se basan en la fuerza bruta y el cálculo de árboles de búsqueda, este proyecto utiliza una **Red Neuronal Recurrente (LSTM)** para predecir el siguiente movimiento más probable, imitando la intuición y el conocimiento posicional de jugadores experimentados.

## 🌟 Características

* **Modelo Predictivo LSTM:** El núcleo del bot es una red neuronal profunda que analiza secuencias de movimientos para predecir la siguiente jugada.
* **Entrenamiento de Alta Calidad:** Entrenado sobre un corpus masivo y filtrado de partidas de grandes maestros y jugadores titulados.
* **API para Jugabilidad:** El modelo se expondrá a través de una API RESTful para facilitar la integración con diferentes interfaces.
* **Interfaz Web V2 (React):** Una interfaz de usuario moderna y premium con soporte para temas, gestión de tiempos y una experiencia de juego fluida.
* **Dashboard de Análisis de IA:** Visualización interactiva en tiempo real del modelo LSTM, mostrando las 10 mejores predicciones y métricas de inferencia.



## 🏗️ Arquitectura del Proyecto

El proyecto está diseñado con una arquitectura moderna de tres componentes principales:

1.  **Modelo de IA (`model.py`):** La red neuronal construida con TensorFlow/Keras que constituye el cerebro del bot (3.48M parámetros).
2.  **Backend (`main.py`):** Un servidor web construido con **FastAPI** desplegado en Hugging Face que expone el modelo mediante una API RESTful.
3.  **Frontend V2 (`frontend-v2`):** Una aplicación de alto rendimiento construida con **React + Vite**, utilizando `Chess.js` para la lógica y `react-chessboard` para la visualización, con un sistema de diseño basado en Glassmorphism.

## 🛠️ Stack Tecnológico

* **Backend:** Python 3.11, FastAPI (Desplegado en Hugging Face Spaces)
* **Machine Learning:** TensorFlow, Keras, NumPy, Scikit-learn
* **Manejo de Lógica de Ajedrez:** `python-chess` / `chess.js`
* **Entorno:** JupyterLab, Google Colab (para entrenamiento con GPU)
* **Frontend:** React 18, Vite, Framer Motion (animaciones), Lucide React (iconos)

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