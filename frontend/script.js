// script.js (Versión Final Ordenada)

// --- 1. VARIABLES GLOBALES ---
var board = null;
var game = new Chess();

// --- 2. FUNCIONES DE LÓGICA DEL JUEGO ---
function onDragStart (source, piece, position, orientation) {
  // No permitir que se muevan piezas si el juego terminó
  if (game.game_over()) return false;

  // Solo permitir que se muevan las piezas del turno actual
  if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
      (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    return false;
  }
}

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

// Actualiza la posición del tablero después de que un movimiento legal se completa
function onSnapEnd () {
  board.position(game.fen());
}

function updateStatus () {
  var status = '';
  var moveColor = 'Blancas';
  if (game.turn() === 'b') {
    moveColor = 'Negras';
  }

  if (game.in_checkmate()) {
    status = 'Juego Terminado, ' + moveColor + ' en Jaque Mate.';
  }
  else if (game.in_draw()) {
    status = 'Juego Terminado, Empate.';
  }
  else {
    status = 'Turno de las ' + moveColor;
    if (game.in_check()) {
      status += ', ' + moveColor + ' están en Jaque.';
    }
  }
  
  // Imprimimos el estado en la consola del navegador
  console.log(status);
}

// --- 3. CONFIGURACIÓN E INICIALIZACIÓN DEL TABLERO ---
// Definimos la configuración UNA SOLA VEZ
var config = {
  draggable: true,
  position: 'start',
  // --- ESTA ES LA LÍNEA QUE CAMBIA ---
  // Ahora apunta a nuestra carpeta local de imágenes
  pieceTheme: 'img/chesspieces/wikipedia/{piece}.png',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
};

// Inicializamos el tablero. Esta línea debe ir DESPUÉS de definir las funciones
board = Chessboard('miTablero', config);
updateStatus();