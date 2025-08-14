// script.js (Con Interfaz de Usuario)

// --- 1. VARIABLES GLOBALES ---
var board = null;
var game = new Chess();
// Referencias a los nuevos elementos del HTML
var statusEl = document.getElementById('status');
var pgnEl = document.getElementById('pgn');

// --- 2. FUNCIONES DE LÓGICA DEL JUEGO ---

function makeRandomMove () {
  var possibleMoves = game.moves();
  if (game.game_over()) return;

  var randomIdx = Math.floor(Math.random() * possibleMoves.length);
  game.move(possibleMoves[randomIdx]);
  
  board.position(game.fen());
  updateStatus();
}

function onDragStart (source, piece, position, orientation) {
  if (game.game_over()) return false;
  if (piece.search(/^b/) !== -1) return false;
}

function onDrop (source, target) {
  var move = game.move({
    from: source,
    to: target,
    promotion: 'q'
  });

  if (move === null) return 'snapback';

  updateStatus();
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
  
  // Actualizamos el contenido de los párrafos en el HTML
  statusEl.innerHTML = status;
  pgnEl.innerHTML = game.pgn();
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

// --- 4. LÓGICA DE LOS BOTONES ---
document.getElementById('resetButton').addEventListener('click', function() {
    game.reset();
    board.start();
    updateStatus();
});