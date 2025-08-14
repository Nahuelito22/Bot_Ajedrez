// script.js (Con bot que juega al azar)

// --- 1. VARIABLES GLOBALES ---
var board = null;
var game = new Chess();

// --- 2. FUNCIONES DE LÓGICA DEL JUEGO ---

// Función para que la computadora haga un movimiento aleatorio
function makeRandomMove () {
  var possibleMoves = game.moves();

  // Si el juego terminó, no hacer nada
  if (game.game_over()) return;

  // Elegir un movimiento al azar de la lista de movimientos legales
  var randomIdx = Math.floor(Math.random() * possibleMoves.length);
  game.move(possibleMoves[randomIdx]);
  
  // Actualizar el tablero visual con el movimiento de la IA
  board.position(game.fen());

  updateStatus();
}

function onDragStart (source, piece, position, orientation) {
  if (game.game_over()) return false;
  // Solo permitir mover piezas blancas
  if (piece.search(/^b/) !== -1) return false;
}

function onDrop (source, target) {
  // Ver si el movimiento del humano es legal
  var move = game.move({
    from: source,
    to: target,
    promotion: 'q'
  });

  if (move === null) return 'snapback';

  updateStatus();

  // Le damos 250 milisegundos a la computadora para "pensar" y responder
  window.setTimeout(makeRandomMove, 250);
}

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
  
  console.log(status);
}

// --- 3. CONFIGURACIÓN E INICIALIZACIÓN DEL TABLERO ---
var config = {
  draggable: true,
  position: 'start',
  pieceTheme: 'img/chesspieces/wikipedia/{piece}.png',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
};

board = Chessboard('miTablero', config);
updateStatus();