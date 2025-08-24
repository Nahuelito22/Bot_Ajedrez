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

    // Elements
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

    // --- HELPERS: PARSE RGB Y LUMINANCE ---
    function parseRGB(rgbString) {
      const m = rgbString.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (!m) return null;
      return [parseInt(m[1], 10), parseInt(m[2], 10), parseInt(m[3], 10)];
    }

    function getLuminance(rgbArray) {
      if (!rgbArray) return 1;
      const [r, g, b] = rgbArray.map(v => v / 255);
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

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
        if (game.in_check()) {
          status += `, ${moveColor} están en Jaque.`;
        }
      }
      statusEl.innerHTML = status;
      pgnEl.innerHTML = game.pgn();
    }

    // --- 3. HIGHLIGHT CONTRASTANTE ---
    function highlightSquare(square) {
        const boardEl = document.getElementById('miTablero');
        if (!boardEl) return;
        const squareEl = boardEl.querySelector(`[data-square='${square}']`);
        if (!squareEl) return;

        // des-resalta previa si existe y distinta
        if (selectedSquare && selectedSquare !== square) {
            unhighlightSquare(selectedSquare);
        }

        if (!squareEl.dataset._origBoxShadow) {
            squareEl.dataset._origBoxShadow = squareEl.style.boxShadow || '';
            squareEl.dataset._origBorderRadius = squareEl.style.borderRadius || '';
        }

        const bg = window.getComputedStyle(squareEl).backgroundColor;
        const rgb = parseRGB(bg);
        const lum = getLuminance(rgb);

        let hlColor;
        if (lum < 0.5) {
            hlColor = 'rgba(255,255,255,0.95)';
        } else {
            hlColor = 'rgba(0,120,0,0.95)';
        }

        squareEl.style.boxShadow = `inset 0 0 0 4px ${hlColor}`;
        squareEl.style.borderRadius = '6px';
    }

    function unhighlightSquare(square) {
        const boardEl = document.getElementById('miTablero');
        if (!boardEl) return;
        const squareEl = boardEl.querySelector(`[data-square='${square}']`);
        if (!squareEl) return;
        if (squareEl.dataset._origBoxShadow !== undefined) {
            squareEl.style.boxShadow = squareEl.dataset._origBoxShadow;
            squareEl.style.borderRadius = squareEl.dataset._origBorderRadius || '';
            delete squareEl.dataset._origBoxShadow;
            delete squareEl.dataset._origBorderRadius;
        } else {
            squareEl.style.boxShadow = '';
            squareEl.style.borderRadius = '';
        }
    }

    // --- 4. DRAG & DROP + TOUCH-BLOCK ---
    function _preventTouchMove(e) { e.preventDefault(); }

    function onDragStart(source, piece, position, orientation) {
        if (game.game_over() || (game.turn() === 'b') || isAiThinking) {
            return false;
        }
        // bloquear scroll en móviles mientras se arrastra
        try {
            document.addEventListener('touchmove', _preventTouchMove, { passive: false });
        } catch (err) {
            document.addEventListener('touchmove', _preventTouchMove);
        }
        document.body.classList.add('body-no-scroll');
        return true;
    }

    function onDrop(source, target) {
        const move = game.move({ from: source, to: target, promotion: 'q' });
        if (move === null) {
            // quitar bloqueo y snapback
            try { document.removeEventListener('touchmove', _preventTouchMove); } catch (e) {}
            document.body.classList.remove('body-no-scroll');
            return 'snapback';
        }
        // movimiento válido
        updateStatus();
        try { document.removeEventListener('touchmove', _preventTouchMove); } catch (e) {}
        document.body.classList.remove('body-no-scroll');
        window.setTimeout(getAiMove, 250);
        return;
    }

    function onSnapEnd() {
        board.position(game.fen());
        try { document.removeEventListener('touchmove', _preventTouchMove); } catch (e) {}
        document.body.classList.remove('body-no-scroll');
    }

    // --- 5. CLICK-TO-MOVE (compatibilidad con drag) ---
    function onSquareClick(square) {
        if (isAiThinking || game.turn() !== 'w') return;

        const pieceOnSquare = game.get(square);

        if (selectedSquare) {
            // Si clic en la misma casilla -> deseleccionar
            if (selectedSquare === square) {
                unhighlightSquare(selectedSquare);
                selectedSquare = null;
                return;
            }

            // Si clic en otra pieza blanca -> cambiar selección
            if (pieceOnSquare && pieceOnSquare.color === 'w') {
                unhighlightSquare(selectedSquare);
                selectedSquare = square;
                highlightSquare(square);
                return;
            }

            // intentar mover desde selectedSquare -> square
            const move = game.move({ from: selectedSquare, to: square, promotion: 'q' });

            if (move === null) {
                // ilegal -> deseleccionar
                unhighlightSquare(selectedSquare);
                selectedSquare = null;
                return;
            }

            // movimiento válido
            unhighlightSquare(selectedSquare);
            selectedSquare = null;
            board.position(game.fen());
            updateStatus();
            window.setTimeout(getAiMove, 250);
        } else {
            // seleccionar si hay pieza blanca
            if (pieceOnSquare && pieceOnSquare.color === 'w') {
                selectedSquare = square;
                highlightSquare(square);
            }
        }
    }

    // --- 6. BOTONES Y MODAL ---
    document.getElementById('resetButton').addEventListener('click', () => {
      game.reset();
      board.start();
      updateStatus();
      // quitar selección visual
      if (selectedSquare) { unhighlightSquare(selectedSquare); selectedSquare = null; }
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

    function getPieceThemePath(themeName) {
        const extension = pieceThemeExtensions[themeName] || 'png';
        return `img/chesspieces/${themeName}/{piece}.${extension}`;
    }

    // modal open/close
    settingsBtn.onclick = () => {
      modal.style.display = "block";
      modal.setAttribute('aria-hidden', 'false');
      // preview
      pieceThemeSelector.value = currentPieceTheme;
      updatePreview(currentPieceTheme);
    };
    function closeModal() {
        modal.style.display = "none";
        modal.setAttribute('aria-hidden', 'true');
        modalContent.removeAttribute('data-board-theme');
    }
    closeBtn.onclick = closeModal;
    window.addEventListener('click', (event) => { if (event.target == modal) closeModal(); });

    confirmThemeBtn.addEventListener('click', () => {
        currentPieceTheme = pieceThemeSelector.value;
        // rebuild board with new piece theme
        const newBoardConfig = Object.assign({}, board.getConfig ? board.getConfig() : {}, { pieceTheme: getPieceThemePath(currentPieceTheme) });
        try { board.destroy(); } catch (e) {}
        board = Chessboard('miTablero', Object.assign({}, newBoardConfig, {
            draggable: true,
            onDragStart, onDrop, onSnapEnd, onSquareClick
        }));
        board.resize();
        // board color theme
        currentBoardColor = modalContent.getAttribute('data-board-theme') || 'default';
        if (currentBoardColor === 'default') {
          document.body.removeAttribute('data-board-theme');
        } else {
          document.body.setAttribute('data-board-theme', currentBoardColor);
        }
        closeModal();
    });

    // preview update function (mini-board)
    function updatePreview(themeName) {
        const previewConfig = {
            position: 'rnbqkbnr/pppppppp/8/8/8/8/8/8 w - - 0 1',
            pieceTheme: getPieceThemePath(themeName)
        };
        try { if (previewBoard) previewBoard.destroy(); } catch (e) {}
        previewBoard = Chessboard('previewBoardPieces', previewConfig);
    }

    // color buttons (preview)
    document.querySelectorAll('.color-btn').forEach(button => {
      button.addEventListener('click', function() {
        const color = this.getAttribute('data-color');
        if (color === 'default') modalContent.removeAttribute('data-board-theme');
        else modalContent.setAttribute('data-board-theme', color);
        // visual de selección
        document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('selected'));
        this.classList.add('selected');
        setPreviewBoardColor(color);
      });
    });

    function setPreviewBoardColor(color) {
      if (color === 'default') {
        modalContent.removeAttribute('data-board-theme');
      } else {
        modalContent.setAttribute('data-board-theme', color);
      }
    }

    // --- 7. THEME SWITCH (dark/light) ---
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

    // --- 8. MENÚ HAMBURGUESA (toggle + aria + cerrar en link click) ---
    hamburgerMenu.addEventListener('click', () => {
        const isActive = hamburgerMenu.classList.toggle('active');
        navLinks.classList.toggle('active');
        document.body.classList.toggle('body-no-scroll');
        hamburgerMenu.setAttribute('aria-expanded', isActive ? 'true' : 'false');
    });

    // Cerrar menu al clickear link
    document.querySelectorAll('.nav-links a').forEach(a => {
        a.addEventListener('click', () => {
            navLinks.classList.remove('active');
            hamburgerMenu.classList.remove('active');
            document.body.classList.remove('body-no-scroll');
            hamburgerMenu.setAttribute('aria-expanded', 'false');
        });
    });

    // cerrar con Escape (modal o nav)
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (modal.style.display === 'block') closeModal();
        if (navLinks.classList.contains('active')) {
          navLinks.classList.remove('active');
          hamburgerMenu.classList.remove('active');
          document.body.classList.remove('body-no-scroll');
          hamburgerMenu.setAttribute('aria-expanded', 'false');
        }
      }
    });

    // --- 9. RESIZE HANDLER (debounced) ---
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => { clearTimeout(timeout); func(...args); };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    const handleResize = debounce(() => {
      try {
        if (board && typeof board.resize === 'function') board.resize();
      } catch (e) { console.warn('board.resize error', e); }
      try { if (previewBoard && typeof previewBoard.resize === 'function') previewBoard.resize(); } catch (e) {}
    }, 200);
    window.addEventListener('resize', handleResize);

    // --- 10. INICIALIZACIÓN DEL TABLERO PRINCIPAL ---
    const boardConfig = {
      draggable: true,
      position: 'start',
      onDragStart,
      onDrop,
      onSnapEnd,
      onSquareClick,
      pieceTheme: getPieceThemePath(currentPieceTheme)
    };
    board = Chessboard('miTablero', boardConfig);
    updateStatus();

});
