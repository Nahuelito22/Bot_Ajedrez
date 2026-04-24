# Roque Chess V1 - Un Bot de Ajedrez con IA ♟️

**[➡️ JUGAR DEMO EN VIVO ⬅️](https://nahuelito22.github.io/Bot_Ajedrez/docs/)**

**Roque Chess** es un bot de ajedrez diseñado para jugar con un estilo "humano", aprendiendo de una base de datos de **1 millón de partidas** de jugadores de alto nivel (2000+ ELO) de la plataforma Lichess.

A diferencia de los motores tradicionales que se basan en el cálculo bruto, este proyecto utiliza una **Red Neuronal Recurrente (LSTM)** para predecir el siguiente movimiento más probable, imitando la intuición y el conocimiento posicional de jugadores experimentados.

![Captura de pantalla del juego](docs/assets/captura_de_pantalla.png) 


---

## 🌟 Características

* **Modelo Predictivo LSTM:** El núcleo del bot es una red neuronal profunda que analiza secuencias de movimientos para predecir la siguiente jugada.
* **Entrenamiento de Alta Calidad:** Entrenado sobre un corpus masivo y filtrado de partidas de grandes maestros y jugadores titulados.
* **API para Jugabilidad:** El modelo se expondrá a través de una API RESTful para facilitar la integración con diferentes interfaces.
* **Interfaz Web V2 (React):** Una interfaz de usuario moderna y premium con soporte para temas, gestión de tiempos y una experiencia de juego fluida.
* **Dashboard de Análisis de IA:** Visualización interactiva en tiempo real del modelo LSTM, mostrando las 10 mejores predicciones y métricas de inferencia.

## 🔬 Análisis del Modelo

Se realizó un análisis exhaustivo del rendimiento del modelo (`modelo_epoch_07.keras`) para determinar su fuerza y estilo de juego.

### Rendimiento General
El bot demuestra un conocimiento de nivel experto en las aperturas y una precisión casi perfecta en los finales, pero es tácticamente vulnerable en el medio juego.

![Gráfico de Rendimiento Promedio](docs/assets/grafico_de_rendimiento.png)

1.  **Modelo de IA (`model.py`):** La red neuronal construida con TensorFlow/Keras que constituye el cerebro del bot (3.48M parámetros).
2.  **Backend (`main.py`):** Un servidor web construido con **FastAPI** desplegado en Hugging Face que expone el modelo mediante una API RESTful.
3.  **Frontend V2 (`frontend-v2`):** Una aplicación de alto rendimiento construida con **React + Vite**, utilizando `Chess.js` para la lógica y `react-chessboard` para la visualización, con un sistema de diseño basado en Glassmorphism.

## 🛠️ Stack Tecnológico

* **Backend:** Python 3.11, FastAPI (Desplegado en Hugging Face Spaces)
* **Machine Learning:** TensorFlow, Keras, NumPy, Scikit-learn
* **Manejo de Lógica de Ajedrez:** `python-chess` / `chess.js`
* **Entorno:** JupyterLab, Google Colab (para entrenamiento con GPU)
* **Frontend:** React 18, Vite, Framer Motion (animaciones), Lucide React (iconos)

---

## 🚀 Instalación y Uso Local

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/Nahuelito22/Bot_Ajedrez.git](https://github.com/Nahuelito22/Bot_Ajedrez.git)
    cd Bot_Ajedrez
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    py -3.12.10 -m venv venv
    venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar los servidores locales:**
    * **Terminal 1 (Backend):** `uvicorn main:app --reload`
    * **Terminal 2 (Frontend):** `cd docs` y luego `py -m http.server 8001`
    * Abrí tu navegador en `http://localhost:8001`.

---

## 🙏 Agradecimientos y Créditos

Este proyecto fue posible gracias a una increíble cantidad de herramientas y comunidades de código abierto:

* **Dataset:** [Lichess Open Database](https://database.lichess.org/) por proveer millones de partidas.
* **Frontend:** [chessboard.js](https://chessboardjs.com/) y [chess.js](https://github.com/jhlywa/chess.js) por las fantásticas librerías para la interfaz.
* **Medición de Elo:** [Cute Chess](https://cutechess.com/) como el gestor de torneos, [Stockfish](https://stockfishchess.org/) como el motor de referencia, y [Ordo](https://github.com/michiguel/Ordo) para el cálculo de ratings.
* **Desarrollo:** Agradecimientos a las comunidades detrás de Python, FastAPI, TensorFlow y todas las librerías que hacen posible la IA moderna.
* **Formación:** Este proyecto fue la entrega final para la carrera de **Data Science** en [Coderhouse](https://www.coderhouse.com/). En primera instancia fue una simple RNN, pero con el tiempo se busco terminarlo y poder hacer un deploy un poco mas serio y robusto.

---

## ✍️ Autor

**Matias Nahuel Ghilardi Salinas**
* **GitHub:** [@Nahuelito22](https://github.com/Nahuelito22)
