// script.js

// Variable para el tablero visual
var board = null; 
// Variable para la lógica del juego
var game = new Chess();
// Variable para mostrar el estado del juego
var statusEl = document.getElementById('status'); 

// Esta función se ejecuta cuando se suelta una pieza
function onDrop (source, target) {
  // Ver si el movimiento es legal
  var move = game.move({
    from: source,
    to: target,
    promotion: 'q' // NOTA: Siempre promocionamos a Reina por simplicidad
  });

  // Si el movimiento es ilegal, volvemos la pieza a su lugar
  if (move === null) return 'snapback';

  updateStatus();
}

// Esta función se llama después de que una pieza se levanta
// No permitimos que se muevan piezas si el juego terminó
function onDragStart (source, piece, position, orientation) {
  if (game.game_over()) return false;

  // Solo permitimos que se muevan las piezas del turno actual
  if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
      (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    return false;
  }
}

// Actualiza la posición del tablero después de un movimiento legal
function onSnapEnd () {
  board.position(game.fen());
}

// Función para actualizar el texto de estado
function updateStatus () {
  var status = '';
  var moveColor = 'Blancas';
  if (game.turn() === 'b') {
    moveColor = 'Negras';
  }

  // Comprobar si hay jaque mate
  if (game.in_checkmate()) {
    status = 'Juego Terminado, ' + moveColor + ' en Jaque Mate.';
  }
  // Comprobar si es un empate
  else if (game.in_draw()) {
    status = 'Juego Terminado, Empate.';
  }
  // El juego continúa
  else {
    status = 'Turno de las ' + moveColor;
    // Comprobar si hay jaque
    if (game.in_check()) {
      status += ', ' + moveColor + ' están en Jaque.';
    }
  }
  
  // No necesitamos un elemento de estado por ahora, solo lo mostramos en la consola
  console.log(status);
}


// Configuración del tablero
var config = {
  draggable: true,
  position: 'start',
  pieceTheme: 'https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/img/chesspieces/wikipedia/{piece}.png',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
};

// Inicializamos el tablero
board = Chessboard('miTablero', config);
updateStatus();