// script.js (Versión Corregida con bloqueo de tablero)

// --- 1. VARIABLES GLOBALES ---
var board = null;
var game = new Chess();
var statusEl = document.getElementById('status');
var pgnEl = document.getElementById('pgn');
const API_URL = "http://127.0.0.1:8000/predict_move";

// --- NUEVA VARIABLE DE CONTROL ---
var isAiThinking = false;


// --- 2. FUNCIONES DE LÓGICA DEL JUEGO ---
async function getAiMove() {
  // Activamos el "interruptor" para bloquear al jugador
  isAiThinking = true;
  statusEl.innerHTML = "El bot está pensando...";

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ moves: game.history() }),
    });

    if (!response.ok) {
      throw new Error(`Error del servidor: ${response.statusText}`);
    }

    const data = await response.json();
    game.move(data.bot_move);
    board.position(game.fen());

  } catch (error) {
    console.error("Error al obtener la jugada del bot:", error);
    statusEl.innerHTML = "Error al conectar con la IA. ¿El servidor está corriendo?";
  } finally {
    // Apagamos el "interruptor" para desbloquear al jugador
    isAiThinking = false;
    updateStatus();
  }
}

function onDragStart (source, piece, position, orientation) {
  // --- LÍNEA MODIFICADA ---
  // No permitir mover si el juego terminó, no es tu turno, O SI LA IA ESTÁ PENSANDO
  if (game.game_over() || game.turn() !== 'w' || isAiThinking) {
    return false;
  }
}

async function onDrop (source, target) {
  var move = game.move({ from: source, to: target, promotion: 'q' });
  if (move === null) return 'snapback';

  updateStatus();
  window.setTimeout(getAiMove, 250);
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
  } else if (game.in_draw()) {
    status = 'Juego Terminado, Empate.';
  } else {
    status = 'Turno de las ' + moveColor;
    if (game.in_check()) {
      status += ', ' + moveColor + ' están en Jaque.';
    }
  }
  
  statusEl.innerHTML = status;
  pgnEl.innerHTML = game.pgn();
}

// --- 3. CONFIGURACIÓN E INICIALIZACIÓN ---
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

// --- 4. LÓGICA DE BOTONES ---
document.getElementById('resetButton').addEventListener('click', function() {
  game.reset();
  board.start();
  updateStatus();
});