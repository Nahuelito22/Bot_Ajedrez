// ========================================================
//     SCRIPT FINAL Y DEFINITIVO (v5 - CON PERSONALIZACIÓN DE PUNTOS)
// ========================================================

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. VARIABLES GLOBALES E INICIALIZACIÓN ---
    var board = null;
    var game = new Chess();
    var isAiThinking = false;
    var currentPieceTheme = 'wikipedia';
    var currentBoardColor = 'default';
    var currentDotColor = 'default'; // <-- NUEVA VARIABLE
    var selectedSquare = null;
    var previewBoard = null;

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

    // Referencias a elementos del DOM
    const boardEl = document.getElementById('miTablero');
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

    // --- 2. LÓGICA DEL BOT (API) ---
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
            const botMoves = data.bot_moves || [];
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
            if (game.in_check()) status += `, ${moveColor} están en Jaque.`;
        }
        statusEl.innerHTML = status;
        pgnEl.innerHTML = game.pgn();
    }

    // --- 3. LÓGICA DE MOVIMIENTOS ---
    boardEl.addEventListener('click', (e) => {
        const squareEl = e.target.closest('[data-square]');
        if (squareEl) {
            const square = squareEl.getAttribute('data-square');
            handleBoardClick(square);
        }
    }, true);

    function handleBoardClick(square) {
        if (isAiThinking || game.turn() !== 'w') return;
        const pieceOnSquare = game.get(square);
        removeMoveDots();
        if(selectedSquare) unhighlightSquare(selectedSquare);
        if (selectedSquare) {
            if (selectedSquare === square) { selectedSquare = null; return; }
            if (pieceOnSquare && pieceOnSquare.color === 'w') {
                selectedSquare = square;
                highlightSquare(square);
                showMoveDots(square);
                return;
            }
            const move = game.move({ from: selectedSquare, to: square, promotion: 'q' });
            if (move === null) { selectedSquare = null; return; }
            board.position(game.fen());
            updateStatus();
            selectedSquare = null;
            window.setTimeout(getAiMove, 250);
        } else {
            if (pieceOnSquare && pieceOnSquare.color === 'w') {
                selectedSquare = square;
                highlightSquare(square);
                showMoveDots(square);
            }
        }
    }

    // --- 4. AYUDAS VISUALES (PUNTOS Y RESALTADO) ---
    function showMoveDots(square) {
        const moves = game.moves({ square: square, verbose: true });
        moves.forEach(move => {
            const dot = document.createElement('div');
            dot.classList.add('move-dot');
            boardEl.querySelector(`[data-square=${move.to}]`).appendChild(dot);
        });
    }

    function removeMoveDots() {
        boardEl.querySelectorAll('.move-dot').forEach(dot => dot.remove());
    }

    function parseRGB(rgbString) {
        const m = rgbString.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        return m ? [parseInt(m[1]), parseInt(m[2]), parseInt(m[3])] : null;
    }

    function getLuminance(rgb) {
        if (!rgb) return 0;
        const [r, g, b] = rgb.map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    function highlightSquare(square) {
        const squareEl = boardEl.querySelector(`[data-square=${square}]`);
        if (!squareEl) return;
        const bg = window.getComputedStyle(squareEl).backgroundColor;
        const lum = getLuminance(parseRGB(bg));
        const highlightColor = lum < 0.5 ? 'rgba(255, 255, 0, 0.8)' : 'rgba(204, 102, 0, 0.8)';
        squareEl.style.boxShadow = `inset 0 0 2px 2px ${highlightColor}`;
    }

    function unhighlightSquare(square) {
        const squareEl = boardEl.querySelector(`[data-square=${square}]`);
        if (squareEl) squareEl.style.boxShadow = '';
    }

    // --- 5. LÓGICA DE BOTONES Y MODAL DE AJUSTES ---
    document.getElementById('resetButton').addEventListener('click', () => {
        game.reset();
        board.start();
        updateStatus();
        if (selectedSquare) { unhighlightSquare(selectedSquare); selectedSquare = null; }
        removeMoveDots();
    });

    document.getElementById('savePgnButton').addEventListener('click', () => {
        navigator.clipboard.writeText(game.pgn()).then(() => {
            const btn = document.getElementById('savePgnButton');
            btn.innerText = '¡Copiado!';
            setTimeout(() => { btn.innerText = 'Copiar Partida (PGN)'; }, 2000);
        });
    });

    function getPieceThemePath(themeName) {
        const extension = pieceThemeExtensions[themeName] || 'svg';
        return `img/chesspieces/${themeName}/{piece}.${extension}`;
    }

    function updatePreview(themeName) {
        const previewConfig = { position: 'rnbqkbnr/pppppppp/8/8/8/8/8/8 w - - 0 1', pieceTheme: getPieceThemePath(themeName) };
        if (previewBoard) previewBoard.destroy();
        previewBoard = Chessboard('previewBoardPieces', previewConfig);
    }

    settingsBtn.onclick = () => {
        modal.style.display = "block";
        // Restaurar selecciones actuales
        pieceThemeSelector.value = currentPieceTheme;
        document.querySelector(`.color-btn[data-color="${currentBoardColor}"]`).classList.add('selected');
        document.querySelector(`.dot-color-btn[data-dot-color="${currentDotColor}"]`).classList.add('selected');
        // Aplicar temas a la previsualización
        modalContent.setAttribute('data-board-theme', currentBoardColor);
        updatePreview(currentPieceTheme);
    };

    function closeModal() { modal.style.display = "none"; }
    closeBtn.onclick = closeModal;
    window.addEventListener('click', (event) => { if (event.target == modal) closeModal(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

    pieceThemeSelector.addEventListener('change', function() { updatePreview(this.value); });

    document.querySelectorAll('.color-btn').forEach(button => {
        button.addEventListener('click', function() {
            document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('selected'));
            this.classList.add('selected');
            modalContent.setAttribute('data-board-theme', this.getAttribute('data-color'));
        });
    });

    // Listener para los nuevos botones de color de puntos
    document.querySelectorAll('.dot-color-btn').forEach(button => {
        button.addEventListener('click', function() {
            document.querySelectorAll('.dot-color-btn').forEach(b => b.classList.remove('selected'));
            this.classList.add('selected');
        });
    });

    confirmThemeBtn.addEventListener('click', () => {
        // Aplicar tema de piezas
        currentPieceTheme = pieceThemeSelector.value;
        
        // Aplicar tema de color del tablero
        const selectedBoardColor = document.querySelector('.color-btn.selected').getAttribute('data-color');
        currentBoardColor = selectedBoardColor;
        document.body.setAttribute('data-board-theme', currentBoardColor);

        // Aplicar tema de color de los puntos
        const selectedDotColor = document.querySelector('.dot-color-btn.selected').getAttribute('data-dot-color');
        currentDotColor = selectedDotColor;
        document.body.setAttribute('data-dot-theme', currentDotColor);

        // Reconstruir tablero
        const newBoardConfig = {
            draggable: false,
            position: game.fen(),
            pieceTheme: getPieceThemePath(currentPieceTheme)
        };
        board.destroy();
        board = Chessboard('miTablero', newBoardConfig);
        board.resize();
        closeModal();
    });

    // --- 6. TEMA OSCURO, MENÚ HAMBURGUESA Y RESIZE ---
    function switchTheme(e) {
        document.body.classList.toggle('dark-theme', e.target.checked);
        localStorage.setItem('theme', e.target.checked ? 'dark' : 'light');
    }
    toggleSwitch.addEventListener('change', switchTheme);
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme === 'dark') { toggleSwitch.checked = true; document.body.classList.add('dark-theme'); }

    hamburgerMenu.addEventListener('click', () => {
        const isActive = navLinks.classList.toggle('active');
        hamburgerMenu.classList.toggle('active');
        document.body.classList.toggle('body-no-scroll', isActive);
    });

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => { clearTimeout(timeout); func(...args); };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    window.addEventListener('resize', debounce(() => { if (board) board.resize(); }, 250));

    // --- 7. INICIALIZACIÓN DEL TABLERO PRINCIPAL ---
    const boardConfig = {
        draggable: false,
        position: 'start',
        pieceTheme: getPieceThemePath(currentPieceTheme)
    };
    board = Chessboard('miTablero', boardConfig);
    updateStatus();
});