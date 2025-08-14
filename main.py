# main.py

from fastapi import FastAPI
import uvicorn

# Creamos una instancia de la aplicación FastAPI
app = FastAPI()

# Definimos una ruta (endpoint) para la raíz de la URL
@app.get("/")
def read_root():
    """Este endpoint es para verificar que la API está funcionando."""
    return {"message": "¡Hola! Soy la API de tu bot de ajedrez."}

if __name__ == "__main__":
    # Esto permite ejecutar el servidor directamente desde el script
    uvicorn.run(app, host="127.0.0.1", port=8000)