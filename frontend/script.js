// Este código se ejecuta cuando la página ha cargado

// Configuración inicial para el tablero
var config = {
  draggable: true,      // Permite que las piezas se puedan arrastrar
  position: 'start'     // 'start' es la posición inicial del ajedrez
};

// Creamos el tablero dentro del div con el id "miTablero"
var board = Chessboard('miTablero', config);