// ========================================================
//         SCRIPT FINAL PARA EL BOT DE AJEDREZ
// ========================================================

// --- 1. VARIABLES GLOBALES E INICIALIZACIÓN ---
var board = null;
var game = new Chess();
var isAiThinking = false;
var currentPieceTheme = 'wikipedia'; // Tema de piezas por defecto
var currentBoardColor = 'default'; // Color del tablero por defecto

// Mapa de temas de piezas y sus extensiones de archivo
const pieceThemeExtensions = {
  "alpha": "svg", "anarcandy": "svg", "caliente": "svg", "california": "svg",
  "cardinal": "svg", "cburnett": "svg", "celtic": "svg", "chess7": "svg",
  "chessnut": "svg", "companion": "svg", "cooke": "svg", "disguised": "svg",
  "dubrovny": "svg", "fantasy": "svg", "firi": "svg", "fresca": "svg",
  "gioco": "svg", "governor": "svg", "horsey": "svg", "icpieces": "svg",
  "kiwen-suwi": "svg", "kosal": "svg", "leipzig": "svg", "letter": "svg",
  "maestro": "svg", "merida": "svg", "monarchy": "svg", "mono": "svg",
  "mpchess": "svg", "pirouetti": "svg", "pixel": "svg", "reillycraig": "svg",
  "rhosgfx": "svg", "riohacha": "svg", "shapes": "svg", "spatial": "svg",
  "staunty": "svg", "tatiana": "svg", "uscf": "png", "wikipedia": "png", "xkcd": "svg"
};

// Referencias a los elementos del HTML
const statusEl = document.getElementById('status');
const pgnEl = document.getElementById('pgn');
const modal = document.getElementById('settingsModal');
const modalContent = document.querySelector('.modal-content');
const settingsBtn = document.getElementById('settingsButton');
const closeBtn = document.querySelector('.close-button');
const toggleSwitch = document.querySelector('#checkbox');
const confirmThemeBtn = document.getElementById('confirmThemeButton');
const pieceThemeSelector = document.getElementById('pieceThemeSelector');
var previewBoard = null;

// URLs de la API
const API_URL = "http://127.0.0.1:8000/predict_move";

// --- 2. FUNCIONES PRINCIPALES DEL JUEGO ---

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
    let moveMade = false;
    for (const move of botMoves) {
      if (game.move(move)) {
        board.position(game.fen());
        moveMade = true;
        break;
      }
    }
    if (!moveMade) {
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

document.getElementById('resetButton').addEventListener('click', () => {
  game.reset();
  board.start();
  updateStatus();
});

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

// --- Lógica para la ventana modal de Ajustes ---

function getPieceThemePath(themeName) {
    const extension = pieceThemeExtensions[themeName] || 'png';
    return `img/chesspieces/${themeName}/{piece}.${extension}`;
}

function updatePreview(themeName) {
    const previewConfig = {
        position: 'rnbqkbnr/pppppppp/8/8/8/8/8/8 w - - 0 1',
        pieceTheme: getPieceThemePath(themeName)
    };
    if (previewBoard) {
        previewBoard.destroy();
    }
    previewBoard = Chessboard('previewBoardPieces', previewConfig);
}

function highlightSelectedColor(color) {
    document.querySelectorAll('.color-btn').forEach(btn => {
        if (btn.getAttribute('data-color') === color) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });
}

function setPreviewBoardColor(color) {
    if (color === 'default') {
        modalContent.removeAttribute('data-board-theme');
    } else {
        modalContent.setAttribute('data-board-theme', color);
    }
}

settingsBtn.onclick = () => {
  modal.style.display = "block";
  pieceThemeSelector.value = currentPieceTheme;
  updatePreview(currentPieceTheme);
  highlightSelectedColor(currentBoardColor);
  setPreviewBoardColor(currentBoardColor);
};

function closeModal() {
    modal.style.display = "none";
    modalContent.removeAttribute('data-board-theme');
}

closeBtn.onclick = closeModal;
window.onclick = (event) => {
  if (event.target == modal) {
    closeModal();
  }
};

pieceThemeSelector.addEventListener('change', function() {
    const selectedTheme = this.value;
    updatePreview(selectedTheme);
});

document.querySelectorAll('.color-btn').forEach(button => {
  button.addEventListener('click', function() {
    const color = this.getAttribute('data-color');
    highlightSelectedColor(color);
    setPreviewBoardColor(color);
  });
});

confirmThemeBtn.addEventListener('click', () => {
    // Aplicar tema de piezas
    const selectedTheme = pieceThemeSelector.value;
    cambiarTema(selectedTheme);

    // Aplicar color del tablero
    currentBoardColor = modalContent.getAttribute('data-board-theme') || 'default';
    if (currentBoardColor === 'default') {
      document.body.removeAttribute('data-board-theme');
    } else {
      document.body.setAttribute('data-board-theme', currentBoardColor);
    }

    closeModal();
});

function cambiarTema(themeName) {
  currentPieceTheme = themeName;
  const newConfig = {
    draggable: true,
    position: game.fen(),
    pieceTheme: getPieceThemePath(themeName),
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd
  };
  board = Chessboard('miTablero', newConfig);
}

// --- Lógica para el cambio de tema (Dark/Light Mode) ---
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
  pieceTheme: getPieceThemePath(currentPieceTheme),
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
};
board = Chessboard('miTablero', config);
updateStatus();