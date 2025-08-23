// script.js (Versión Final Corregida)

// --- 1. VARIABLES GLOBALES ---
var board = null;
var game = new Chess();
var statusEl = document.getElementById('status');
var pgnEl = document.getElementById('pgn');
const API_URL = "http://127.0.0.1:8000/predict_move";
var isAiThinking = false;

// --- 2. FUNCIONES DE LÓGICA DEL JUEGO ---
async function getAiMove() {
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
    const botMoves = data.bot_moves;
    console.log("La IA propone (en orden):", botMoves);
    var moveMade = false;
    for (const move of botMoves) {
        if (game.move(move)) {
            console.log("Jugada legal encontrada y ejecutada:", move);
            board.position(game.fen());
            moveMade = true;
            break;
        } else {
            console.warn("La jugada propuesta '" + move + "' fue ilegal o inválida. Intentando la siguiente.");
        }
    }
    if (!moveMade) {
        console.error("¡PLAN C ACTIVADO! La IA no dio jugadas legales. Jugando al azar.");
        var possibleMoves = game.moves();
        if (possibleMoves.length > 0) {
            var randomIdx = Math.floor(Math.random() * possibleMoves.length);
            game.move(possibleMoves[randomIdx]);
            board.position(game.fen());
        }
    }
  } catch (error) {
    console.error("Error al obtener la jugada del bot:", error);
    statusEl.innerHTML = "Error al conectar con la IA.";
  } finally {
    isAiThinking = false;
    updateStatus();
  }
}

function onDragStart (source, piece, position, orientation) {
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

function cambiarTema(themeName) {
  // Para cambiar el tema, creamos una nueva configuración
  // y re-inicializamos el tablero. Usamos game.fen() para mantener la posición.
  var newConfig = {
      draggable: true,
      position: game.fen(), // <-- Mantiene la posición actual de la partida
      pieceTheme: `img/chesspieces/${themeName}/{piece}.png`,
      onDragStart: onDragStart,
      onDrop: onDrop,
      onSnapEnd: onSnapEnd
  };
  board = Chessboard('miTablero', newConfig);
}

document.getElementById('wikiButton').addEventListener('click', function() {
  cambiarTema('wikipedia');
});
document.getElementById('alphaButton').addEventListener('click', function() {
  cambiarTema('alpha');
});
document.getElementById('uscfButton').addEventListener('click', function() {
  cambiarTema('uscf');
});


// --- 5. LÓGICA PARA EL CAMBIO DE TEMA (DARK/LIGHT MODE) ---

const toggleSwitch = document.querySelector('#checkbox');

// Función que cambia el tema
function switchTheme(e) {
    if (e.target.checked) {
        document.body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark'); // Guardamos la preferencia
    } else {
        document.body.classList.remove('dark-theme');
        localStorage.setItem('theme', 'light'); // Guardamos la preferencia
    }    
}

// Event listener para el interruptor
toggleSwitch.addEventListener('change', switchTheme);

// Comprobar si el usuario ya tiene una preferencia guardada
const currentTheme = localStorage.getItem('theme');
if (currentTheme) {
    if (currentTheme === 'dark') {
        toggleSwitch.checked = true;
        document.body.classList.add('dark-theme');
    }
}

// --- NUEVO CÓDIGO PARA GUARDAR LA PARTIDA ---

document.getElementById('savePgnButton').addEventListener('click', function() {
    // Obtenemos el historial de la partida en formato PGN
    const pgn = game.pgn();
    
    // Usamos la API del navegador para copiar el texto al portapapeles
    navigator.clipboard.writeText(pgn).then(function() {
        // Éxito: avisamos al usuario que se copió
        const originalText = document.getElementById('savePgnButton').innerText;
        document.getElementById('savePgnButton').innerText = '¡Copiado!';
        
        // Volvemos al texto original después de 2 segundos
        setTimeout(function() {
            document.getElementById('savePgnButton').innerText = originalText;
        }, 2000);

    }, function(err) {
        // Error: por si falla el copiado
        console.error('Error al copiar el PGN: ', err);
        alert("No se pudo copiar la partida.");
    });
});