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
// Reemplaza solo esta función en tu script.js

async function getAiMove() {
  isAiThinking = true;
  statusEl.innerHTML = "El bot está pensando...";
  console.log("--- INTENTANDO OBTENER JUGADA DE LA IA ---");
  console.log("Estado del juego (FEN) antes de la llamada:", game.fen());

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
    const botMove = data.bot_move;
    console.log("IA responde con la jugada:", botMove);

    // --- PUNTO CRÍTICO DE DEPURACIÓN ---
    console.log("Intentando ejecutar game.move('" + botMove + "')");
    var moveResult = game.move(botMove);

    // Verificamos qué devolvió la función move
    if (moveResult === null) {
      console.error("¡ERROR! chess.js rechazó la jugada:", botMove);
      console.log("Lista de movimientos legales posibles según chess.js:", game.moves());
    } else {
      console.log("¡ÉXITO! La jugada fue aceptada por chess.js.");
    }
    
    board.position(game.fen());
    console.log("Estado del juego (FEN) después de la jugada:", game.fen());

  } catch (error) {
    console.error("Error en la comunicación con la IA:", error);
  } finally {
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