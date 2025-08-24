// ========================================================
//         SCRIPT FINAL PARA EL BOT DE AJEDREZ
// ========================================================

// --- 1. VARIABLES GLOBALES E INICIALIZACIÓN ---
var board = null;
var game = new Chess();
var isAiThinking = false;

// Referencias a los elementos del HTML
const statusEl = document.getElementById('status');
const pgnEl = document.getElementById('pgn');
const modal = document.getElementById('settingsModal');
const settingsBtn = document.getElementById('settingsButton');
const closeBtn = document.querySelector('.close-button');
const toggleSwitch = document.querySelector('#checkbox');
var previewBoard = null;

// URLs de la API (Descomentar la que se quiera usar)
const API_URL = "https://nahuelito22-bot-ajedrez.hf.space/predict_move"; // Para producción


// --- 2. FUNCIONES PRINCIPALES DEL JUEGO ---

/**
 * Se comunica con la API del backend para obtener la jugada del bot.
 */
async function getAiMove() {
  isAiThinking = true;
  statusEl.innerHTML = "El bot está pensando...";

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ moves: game.history() }),
    });

    if (!response.ok) throw new Error(`Error del servidor: ${response.statusText}`);

    const data = await response.json();
    const botMoves = data.bot_moves;
    console.log("La IA propone (en orden):", botMoves);

    let moveMade = false;
    for (const move of botMoves) {
      if (game.move(move)) {
        console.log("Jugada legal encontrada y ejecutada:", move);
        board.position(game.fen());
        moveMade = true;
        break;
      } else {
        console.warn(`La jugada propuesta '${move}' fue ilegal. Intentando la siguiente.`);
      }
    }

    if (!moveMade) {
      console.error("¡PLAN C ACTIVADO! La IA no dio jugadas legales. Jugando al azar.");
      const possibleMoves = game.moves();
      if (possibleMoves.length > 0) {
        const randomIdx = Math.floor(Math.random() * possibleMoves.length);
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

/**
 * Actualiza el texto de estado y el PGN en la página.
 */
function updateStatus() {
  let status = '';
  const moveColor = (game.turn() === 'b') ? 'Negras' : 'Blancas';

  if (game.in_checkmate()) {
    status = `Juego Terminado, ${moveColor} en Jaque Mate.`;
  } else if (game.in_draw()) {
    status = 'Juego Terminado, Empate.';
  } else {
    status = `Turno de las ${moveColor}`;
    if (game.in_check()) {
      status += `, ${moveColor} están en Jaque.`;
    }
  }
  
  statusEl.innerHTML = status;
  pgnEl.innerHTML = game.pgn();
}


// --- 3. FUNCIONES DE INTERACCIÓN CON EL TABLERO (CALLBACKS) ---

function onDragStart(source, piece, position, orientation) {
  return !(game.game_over() || game.turn() !== 'w' || isAiThinking);
}

function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: 'q' });
  if (move === null) return 'snapback';
  updateStatus();
  window.setTimeout(getAiMove, 250);
}

function onSnapEnd() {
  board.position(game.fen());
}


// --- 4. LÓGICA DE BOTONES Y AJUSTES ---

// Botón de reiniciar
document.getElementById('resetButton').addEventListener('click', () => {
  game.reset();
  board.start();
  updateStatus();
});

// Botón de copiar PGN
document.getElementById('savePgnButton').addEventListener('click', () => {
  const pgn = game.pgn();
  navigator.clipboard.writeText(pgn).then(() => {
    const btn = document.getElementById('savePgnButton');
    const originalText = btn.innerText;
    btn.innerText = '¡Copiado!';
    setTimeout(() => { btn.innerText = originalText; }, 2000);
  }, (err) => {
    console.error('Error al copiar el PGN: ', err);
  });
});

// Lógica para la ventana modal de Ajustes
settingsBtn.onclick = () => {
  modal.style.display = "block";
  if (!previewBoard) {
    previewBoard = Chessboard('previewBoardPieces', {
      position: 'start',
      pieceTheme: 'img/chesspieces/wikipedia/{piece}.png'
    });
  }
};

closeBtn.onclick = () => { modal.style.display = "none"; };
window.onclick = (event) => {
  if (event.target == modal) {
    modal.style.display = "none";
  }
};

function cambiarTema(themeName) {
  const newConfig = {
    draggable: true,
    position: game.fen(),
    pieceTheme: `img/chesspieces/${themeName}/{piece}.png`,
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd
  };
  board = Chessboard('miTablero', newConfig);
  if (previewBoard) {
    // La librería no tiene un método directo para cambiar el tema, 
    // así que lo reconstruimos también.
    previewBoard = Chessboard('previewBoardPieces', {
        position: 'start',
        pieceTheme: `img/chesspieces/${themeName}/{piece}.png`
    });
  }
}

document.querySelectorAll('.theme-btn').forEach(button => {
  button.addEventListener('click', function() {
    const theme = this.getAttribute('data-theme');
    cambiarTema(theme);
  });
});

// Lógica para el cambio de tema (Dark/Light Mode)
function switchTheme(e) {
  if (e.target.checked) {
    document.body.classList.add('dark-theme');
    localStorage.setItem('theme', 'dark');
  } else {
    document.body.classList.remove('dark-theme');
    localStorage.setItem('theme', 'light');
  }
}

toggleSwitch.addEventListener('change', switchTheme);
const currentTheme = localStorage.getItem('theme');
if (currentTheme === 'dark') {
  toggleSwitch.checked = true;
  document.body.classList.add('dark-theme');
}


// --- 5. INICIALIZACIÓN DEL TABLERO PRINCIPAL ---
const config = {
  draggable: true,
  position: 'start',
  pieceTheme: 'img/chesspieces/wikipedia/{piece}.png',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
};
board = Chessboard('miTablero', config);
updateStatus();