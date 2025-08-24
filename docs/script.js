// ========================================================
//         SCRIPT FINAL PARA EL BOT DE AJEDREZ
// ========================================================

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. VARIABLES GLOBALES E INICIALIZACIÓN ---
    var board = null;
    var game = new Chess();
    var isAiThinking = false;
    var currentPieceTheme = 'wikipedia';
    var currentBoardColor = 'default';
    var selectedSquare = null;

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
    const hamburgerMenu = document.querySelector('.hamburger-menu');
    const navLinks = document.querySelector('.nav-links');
    var previewBoard = null;

    // --- 2. FUNCIONES PRINCIPALES DEL JUEGO ---
    async function getAiMove() {
      isAiThinking = true;
      statusEl.innerHTML = "El bot está pensando...";
      try {
        const response = await fetch(config.API_URL, {
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

    // --- 3. LÓGICA DE CLICK-TO-MOVE Y DRAG-AND-DROP ---
    function onDragStart(source, piece) {
        if (game.game_over() || (game.turn() === 'b') || isAiThinking) {
            return false;
        }
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

    settingsBtn.onclick = () => {
      modal.style.display = "block";
    };

    function closeModal() {
        modal.style.display = "none";
    }

    closeBtn.onclick = closeModal;
    window.onclick = (event) => {
      if (event.target == modal) {
        closeModal();
      }
    };

    confirmThemeBtn.addEventListener('click', () => {
        const selectedTheme = pieceThemeSelector.value;
        currentPieceTheme = selectedTheme;
        const newBoardConfig = {
            draggable: true,
            position: game.fen(),
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            pieceTheme: getPieceThemePath(currentPieceTheme)
        };
        board = Chessboard('miTablero', newBoardConfig);

        currentBoardColor = modalContent.getAttribute('data-board-theme') || 'default';
        if (currentBoardColor === 'default') {
          document.body.removeAttribute('data-board-theme');
        } else {
          document.body.setAttribute('data-board-theme', currentBoardColor);
        }
        closeModal();
    });

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

    // --- 5. LÓGICA DEL MENÚ HAMBURGUESA Y RESIZE ---
    hamburgerMenu.addEventListener('click', () => {
        hamburgerMenu.classList.toggle('active');
        navLinks.classList.toggle('active');
        document.body.classList.toggle('body-no-scroll');
    });

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    const handleResize = debounce(() => {
        if (board) {
            board.resize();
        }
    }, 250);

    window.addEventListener('resize', handleResize);

    // --- 6. INICIALIZACIÓN DEL TABLERO PRINCIPAL ---
    const boardConfig = {
      draggable: true,
      position: 'start',
      onDragStart: onDragStart,
      onDrop: onDrop,
      onSnapEnd: onSnapEnd,
      pieceTheme: getPieceThemePath(currentPieceTheme)
    };
    board = Chessboard('miTablero', boardConfig);
    updateStatus();
});