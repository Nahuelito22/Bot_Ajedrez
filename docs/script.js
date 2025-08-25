// ========================================================
//     SCRIPT FINAL Y DEFINITIVO (v8 - PREVIEW SOLO + RELOJ FIABLE)
// ========================================================

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. VARIABLES GLOBALES E INICIALIZACIÓN ---
    var board = null;
    var game = new Chess();
    var isAiThinking = false;
    var selectedSquare = null;
    var previewBoard = null;

    // Variables de personalización
    var currentPieceTheme = 'wikipedia';
    var currentBoardColor = 'default';
    var currentDotColor = 'default';
    
    // --- NUEVAS VARIABLES DEL RELOJ ---
    var timeControlType = 'unlimited'; // 'unlimited' o 'custom'
    var customMinutes = 10;
    var customSeconds = 0;
    var customIncrement = 0;
    var whiteTime, blackTime, increment;
    var activeClock = null;
    var timerIntervalId = null;

    // Marca temporal del inicio del turno (ms desde epoch)
    var turnStartTimestamp = null;

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
    const whiteClockEl = document.querySelector('.player-clock');
    const blackClockEl = document.querySelector('.opponent-clock');
    const modal = document.getElementById('settingsModal');
    const modalContent = document.querySelector('.modal-content');
    const settingsBtn = document.getElementById('settingsButton');
    const closeBtn = document.querySelector('.close-button');
    const toggleSwitch = document.querySelector('#checkbox');
    const confirmThemeBtn = document.getElementById('confirmThemeButton');
    const pieceThemeSelector = document.getElementById('pieceThemeSelector');
    const timeControlTypeRadios = document.querySelectorAll('input[name="timeControlType"]');
    const customTimeInputsEl = document.querySelector('.custom-time-inputs');
    const timeMinutesInput = document.getElementById('timeMinutes');
    const timeSecondsInput = document.getElementById('timeSeconds');
    const timeIncrementInput = document.getElementById('timeIncrement');
    const hamburgerMenu = document.querySelector('.hamburger-menu');
    const navLinks = document.querySelector('.nav-links');

    // --- 2. LÓGICA DEL RELOJ ---
    function formatTime(ms) {
        if (timeControlType === 'unlimited') return "--:--";
        const totalSeconds = Math.ceil(ms / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    function updateClockDisplays() {
        whiteClockEl.textContent = formatTime(whiteTime);
        blackClockEl.textContent = formatTime(blackTime);
    }

    function stopTimer() {
        if (timerIntervalId) clearInterval(timerIntervalId);
        timerIntervalId = null;
        activeClock = null;
        whiteClockEl.classList.remove('active-clock');
        blackClockEl.classList.remove('active-clock');
    }

    function startTimer() {
        if (timerIntervalId || timeControlType === 'unlimited' || game.game_over()) return;

        activeClock = game.turn(); // 'w' o 'b'
        whiteClockEl.classList.toggle('active-clock', activeClock === 'w');
        blackClockEl.classList.toggle('active-clock', activeClock === 'b');

        // marca del inicio del turno (para medir duración del mismo)
        turnStartTimestamp = Date.now();

        let lastTime = Date.now();
        timerIntervalId = setInterval(() => {
            const now = Date.now();
            // limitar delta para evitar decrementos gigantes tras sleep/hibernación
            let delta = now - lastTime;
            if (delta > 1000) delta = 1000;
            lastTime = now;
            let timeup = false;

            if (activeClock === 'w') {
                if (!isFinite(whiteTime)) return;
                whiteTime -= delta;
                if (whiteTime <= 0) { whiteTime = 0; timeup = true; }
            } else {
                if (!isFinite(blackTime)) return;
                blackTime -= delta;
                if (blackTime <= 0) { blackTime = 0; timeup = true; }
            }
            updateClockDisplays();
            if (timeup) {
                stopTimer();
                alert(`¡Se acabó el tiempo! Ganan las ${activeClock === 'w' ? 'Blancas' : 'Negras'}.`);
                // limpiar tablero
                game.load('8/8/8/8/8/8/8/8 w - - 0 1');
                if (board) board.position(game.fen());
                updateStatus();
            }
        }, 100);
    }

    function switchActiveClock() {
        if (timeControlType === 'unlimited') return;

        // Parar temporizador actual
        stopTimer();

        // Determinar quién acaba de mover
        const justMoved = game.turn() === 'b' ? 'w' : 'b';

        // Calcular duración del turno del que acaba de mover
        let moveDuration = null;
        if (turnStartTimestamp) {
            moveDuration = Date.now() - turnStartTimestamp; // ms
        }

        // Si hay increment configurado, aplicarlo solo si la duración del movimiento
        // fue menor o igual al "lapso" (interpretamos el lapso == customIncrement segundos)
        if (increment > 0 && moveDuration !== null) {
            const windowMs = customIncrement * 1000;
            if (moveDuration <= windowMs) {
                if (justMoved === 'w') whiteTime += increment;
                else blackTime += increment;
            }
        }

        updateClockDisplays();

        // Iniciar el siguiente reloj (startTimer marca de inicio del turno)
        startTimer();
    }

    function initClocks() {
        stopTimer();
        if (timeControlType === 'custom') {
            whiteTime = (customMinutes * 60 + customSeconds) * 1000;
            blackTime = (customMinutes * 60 + customSeconds) * 1000;
            increment = customIncrement * 1000;
        } else {
            whiteTime = Infinity;
            blackTime = Infinity;
            increment = 0;
        }
        // reset turn start marker
        turnStartTimestamp = null;
        updateClockDisplays();
    }

    // --- 3. LÓGICA DEL BOT (API) ---
    async function getAiMove() {
        isAiThinking = true;
        statusEl.innerHTML = "El bot está pensando...";
        try {
            const response = await fetch(config.API_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ moves: game.history() }) });
            if (!response.ok) throw new Error(`Error del servidor: ${response.statusText}`);
            const data = await response.json();
            const botMoves = data.bot_moves || [];
            let moveMade = false;
            for (const move of botMoves) {
                if (game.move(move)) {
                    if (board) board.position(game.fen());
                    moveMade = true;
                    break;
                }
            }
            if (!moveMade) {
                const possibleMoves = game.moves();
                if (possibleMoves.length > 0) {
                    const randomIdx = Math.floor(Math.random() * possibleMoves.length);
                    game.move(possibleMoves[randomIdx]);
                    if (board) board.position(game.fen());
                }
            }
        } catch (error) {
            console.error("Error al obtener la jugada del bot:", error);
            statusEl.innerHTML = "Error al conectar con la IA.";
        } finally {
            isAiThinking = false;
            updateStatus();
            if (!game.game_over()) switchActiveClock();
            else stopTimer();
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

    // --- 4. LÓGICA DE MOVIMIENTOS ---
    boardEl.addEventListener('click', (e) => {
        const squareEl = e.target.closest('[data-square]');
        if (squareEl) {
            const square = squareEl.getAttribute('data-square');
            handleBoardClick(square);
        }
    }, true);

    function handleBoardClick(square) {
        if (isAiThinking || game.turn() !== 'w') return;
        if (game.history().length === 0 && timeControlType === 'custom') startTimer(); // Iniciar reloj en la primera jugada
        const pieceOnSquare = game.get(square);
        removeMoveDots();
        if(selectedSquare) unhighlightSquare(selectedSquare);
        if (selectedSquare) {
            if (selectedSquare === square) { selectedSquare = null; return; }
            if (pieceOnSquare && pieceOnSquare.color === 'w') { selectedSquare = square; highlightSquare(square); showMoveDots(square); return; }
            const move = game.move({ from: selectedSquare, to: square, promotion: 'q' });
            if (move === null) { selectedSquare = null; return; }
            if (board) board.position(game.fen());
            updateStatus();
            selectedSquare = null;
            if (!game.game_over()) {
                switchActiveClock();
                window.setTimeout(getAiMove, 250);
            } else {
                stopTimer();
            }
        } else {
            if (pieceOnSquare && pieceOnSquare.color === 'w') {
                selectedSquare = square;
                highlightSquare(square);
                showMoveDots(square);
            }
        }
    }

    // --- 5. AYUDAS VISUALES (PUNTOS Y RESALTADO) ---
    function showMoveDots(square) {
        const moves = game.moves({ square: square, verbose: true });
        moves.forEach(move => {
            const dot = document.createElement('div');
            dot.classList.add('move-dot');
            const target = boardEl.querySelector(`[data-square="${move.to}"]`);
            if (target) target.appendChild(dot);
        });
    }
    function removeMoveDots() { boardEl.querySelectorAll('.move-dot').forEach(dot => dot.remove()); }
    function parseRGB(rgbString) { const m = rgbString.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/); return m ? [parseInt(m[1]), parseInt(m[2]), parseInt(m[3])] : null; }
    function getLuminance(rgb) { if (!rgb) return 0; const [r, g, b] = rgb.map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }); return 0.2126 * r + 0.7152 * g + 0.0722 * b; }
    function highlightSquare(square) { const squareEl = boardEl.querySelector(`[data-square="${square}"]`); if (!squareEl) return; const bg = window.getComputedStyle(squareEl).backgroundColor; const rgb = parseRGB(bg); const lum = getLuminance(rgb); const highlightColor = lum < 0.5 ? 'rgba(255, 255, 0, 0.8)' : 'rgba(204, 102, 0, 0.8)'; squareEl.style.boxShadow = `inset 0 0 2px 2px ${highlightColor}`; }
    function unhighlightSquare(square) { const squareEl = boardEl.querySelector(`[data-square="${square}"]`); if (squareEl) squareEl.style.boxShadow = ''; }

    // --- 6. LÓGICA DE BOTONES Y MODAL DE AJUSTES ---
    function startNewGame() {
        game.reset();
        if (board) {
            try { board.destroy(); } catch(e){ /* ignore */ }
        }
        const boardConfig = {
            draggable: false,
            position: 'start',
            pieceTheme: getPieceThemePath(currentPieceTheme)
        };
        board = Chessboard('miTablero', boardConfig);
        updateStatus();
        if (selectedSquare) { unhighlightSquare(selectedSquare); selectedSquare = null; }
        removeMoveDots();
        initClocks();
    }

    document.getElementById('resetButton').addEventListener('click', startNewGame);
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
        const previewConfig = {
            position: 'rnbqkbnr/pppppppp/8/8/8/8/8/8 w - - 0 1',
            pieceTheme: getPieceThemePath(themeName)
        };
        if (previewBoard) {
            try { previewBoard.destroy(); } catch(e) {}
        }
        previewBoard = Chessboard('previewBoardPieces', previewConfig);
    }

    settingsBtn.onclick = () => {
        modal.style.display = "block";
        // Restaurar selecciones actuales
        const timeRadio = document.querySelector(`input[name="timeControlType"][value="${timeControlType}"]`);
        if (timeRadio) timeRadio.checked = true;
        if (timeControlType === 'custom') customTimeInputsEl.classList.remove('hidden');
        else customTimeInputsEl.classList.add('hidden');
        timeMinutesInput.value = customMinutes;
        timeSecondsInput.value = customSeconds;
        timeIncrementInput.value = customIncrement;

        // piece theme select
        pieceThemeSelector.value = currentPieceTheme;

        // limpiar clases selected y luego marcar la actual, con guards
        document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('selected'));
        const boardColorBtn = document.querySelector(`.color-btn[data-color="${currentBoardColor}"]`);
        if (boardColorBtn) boardColorBtn.classList.add('selected');

        document.querySelectorAll('.dot-color-btn').forEach(b => b.classList.remove('selected'));
        const dotColorBtn = document.querySelector(`.dot-color-btn[data-dot-color="${currentDotColor}"]`);
        if (dotColorBtn) dotColorBtn.classList.add('selected');

        modalContent.setAttribute('data-board-theme', currentBoardColor);
        updatePreview(currentPieceTheme);
    };

    timeControlTypeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'custom') customTimeInputsEl.classList.remove('hidden');
            else customTimeInputsEl.classList.add('hidden');
        });
    });

    function closeModal() { modal.style.display = "none"; }
    closeBtn.onclick = closeModal;
    window.addEventListener('click', (event) => { if (event.target == modal) closeModal(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

    // Nota importante: SOLO actualizamos la PREVIEW al cambiar el selector.
    // No aplicamos el tema al tablero principal hasta "Aplicar y Jugar".
    pieceThemeSelector.addEventListener('change', function() {
        updatePreview(this.value);
    });

    document.querySelectorAll('.color-btn').forEach(button => {
        button.addEventListener('click', function() {
            document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('selected'));
            this.classList.add('selected');
            modalContent.setAttribute('data-board-theme', this.getAttribute('data-color'));
        });
    });

    document.querySelectorAll('.dot-color-btn').forEach(button => {
        button.addEventListener('click', function() {
            document.querySelectorAll('.dot-color-btn').forEach(b => b.classList.remove('selected'));
            this.classList.add('selected');
        });
    });

    confirmThemeBtn.addEventListener('click', () => {
        // Guardar todos los ajustes
        currentPieceTheme = pieceThemeSelector.value;
        const selBoard = document.querySelector('.color-btn.selected');
        if (selBoard) currentBoardColor = selBoard.getAttribute('data-color');
        const selDot = document.querySelector('.dot-color-btn.selected');
        if (selDot) currentDotColor = selDot.getAttribute('data-dot-color');
        timeControlType = document.querySelector('input[name="timeControlType"]:checked').value;
        customMinutes = parseInt(timeMinutesInput.value, 10) || 0;
        customSeconds = parseInt(timeSecondsInput.value, 10) || 0;
        customIncrement = parseInt(timeIncrementInput.value, 10) || 0;

        // Aplicar temas visuales (board color y dot color)
        document.body.setAttribute('data-board-theme', currentBoardColor);
        document.body.setAttribute('data-dot-theme', currentDotColor);

        // Iniciar nueva partida para aplicar todos los cambios de tema y tiempo
        startNewGame();
        closeModal();
    });

    // --- 7. TEMA OSCURO, MENÚ HAMBURGUESA Y RESIZE ---
    function switchTheme(e) { document.body.classList.toggle('dark-theme', e.target.checked); localStorage.setItem('theme', e.target.checked ? 'dark' : 'light'); }
    toggleSwitch.addEventListener('change', switchTheme);
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme === 'dark') { toggleSwitch.checked = true; document.body.classList.add('dark-theme'); }
    if (hamburgerMenu) {
        hamburgerMenu.addEventListener('click', () => {
            const isActive = navLinks.classList.toggle('active');
            hamburgerMenu.classList.toggle('active');
            document.body.classList.toggle('body-no-scroll', isActive);
        });
    }
    function debounce(func, wait) { let timeout; return function executedFunction(...args) { const later = () => { clearTimeout(timeout); func(...args); }; clearTimeout(timeout); timeout = setTimeout(later, wait); }; }
    window.addEventListener('resize', debounce(() => { if (board) board.resize(); }, 250));

    // --- 8. INICIALIZACIÓN ---
    startNewGame();
});
