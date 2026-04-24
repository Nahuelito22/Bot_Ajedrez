import { useState, useRef, useEffect } from 'react';
import { Chessboard } from 'react-chessboard';
import { RotateCcw, Undo2, Settings, Clock, Copy, User, Bot, Users } from 'lucide-react';
import { useChessGame } from '../hooks/useChessGame';
import TimeModal from './TimeModal';

export default function GameSection({ boardTheme, pieceTheme, onOpenSettings }) {
    const {
        fen,
        isThinking,
        status,
        gameHistory,
        onDrop,
        onSquareClick,
        optionSquares,
        resetGame,
        undoMove,
        getPGN,
        isGameOver,
        gameMode, setGameMode,
        timeControl, setTimeControl,
        increment, setIncrement,
        whiteTime, setWhiteTime,
        blackTime, setBlackTime,
        isTimerRunning
    } = useChessGame();

    const [boardWidth, setBoardWidth] = useState(480);
    const [isTimeModalOpen, setIsTimeModalOpen] = useState(false);
    const boardWrapperRef = useRef();

    useEffect(() => {
        const handleResize = () => {
            if (boardWrapperRef.current) {
                setBoardWidth(boardWrapperRef.current.offsetWidth);
            }
        };

        handleResize();
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Formatear tiempo
    const formatTime = (seconds) => {
        if (timeControl === 'unlimited') return "--:--";
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    // Copiar PGN
    const handleCopyPGN = () => {
        const pgn = getPGN();
        navigator.clipboard.writeText(pgn);
        alert('Partida (PGN) copiada al portapapeles!');
    };

    // Manejar cambio de modo
    const toggleGameMode = () => {
        const newMode = gameMode === 'pvai' ? 'pvp' : 'pvai';
        setGameMode(newMode);
        resetGame(); // Reiniciar partida al cambiar modo
    };

    // Colores por defecto si no hay tema seleccionado
    const lightSquareStyle = boardTheme ? { backgroundColor: boardTheme.light } : { backgroundColor: '#f0d9b5' };
    const darkSquareStyle = boardTheme ? { backgroundColor: boardTheme.dark } : { backgroundColor: '#b58863' };

    // Custom pieces
    const pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK'];
    const customPieces = {};
    pieces.forEach(p => {
        customPieces[p] = ({ squareWidth }) => (
            <div style={{ width: squareWidth, height: squareWidth, backgroundImage: `url(/chesspieces/${pieceTheme || 'wikipedia'}/${p}.${['alpha', 'uscf', 'wikipedia'].includes(pieceTheme) && pieceTheme !== 'wikipedia' ? 'svg' : pieceTheme === 'wikipedia' ? 'png' : 'svg'})`, backgroundSize: '100%', backgroundRepeat: 'no-repeat' }} />
        );
    });

    return (
        <section id="jugar" className="main-section">
            <div className="game-container">
                
                <div className="left-panel">
                    <div className="player-info opponent-info glass-card">
                        <div className="avatar opponent-avatar">
                            {gameMode === 'pvai' ? <Bot size={24} color="#cfb168"/> : <User size={24} color="#cfb168"/>}
                        </div>
                        <div className="player-details">
                            <h3>{gameMode === 'pvai' ? 'Roque Chess AI' : 'Jugador 2 (Negras)'}</h3>
                            <p>{gameMode === 'pvai' ? 'Evaluación Posicional Avanzada' : 'Humano'}</p>
                            {isThinking && <div className="thinking-indicator">Pensando...</div>}
                        </div>
                        <div className={`clock-display ${isTimerRunning && (fen.split(' ')[1] === 'b') ? 'active-clock' : ''}`}>
                            {formatTime(blackTime)}
                        </div>
                    </div>

                    <div className="board-wrapper" ref={boardWrapperRef}>
                        <Chessboard 
                            id="RoqueChessBoard" 
                            position={fen} 
                            onPieceDrop={onDrop}
                            onSquareClick={onSquareClick}
                            customSquareStyles={optionSquares}
                            boardOrientation="white"
                            customDarkSquareStyle={darkSquareStyle}
                            customLightSquareStyle={lightSquareStyle}
                            customPieces={pieceTheme ? customPieces : undefined}
                            animationDuration={200}
                            boardWidth={boardWidth}
                        />
                    </div>

                    <div className="player-info user-info glass-card">
                        <div className="avatar user-avatar">
                            <User size={24} color="#cfb168"/>
                        </div>
                        <div className="player-details">
                            <h3>Jugador 1 (Blancas)</h3>
                            <p>Tú</p>
                        </div>
                        <div className={`clock-display ${isTimerRunning && (fen.split(' ')[1] === 'w') ? 'active-clock' : ''}`}>
                            {formatTime(whiteTime)}
                        </div>
                    </div>
                </div>

                <div className="right-panel">
                    <div className="status-box glass-card">
                        <h3>Estado de la Partida</h3>
                        <p id="gameStatus" className="status-text">{status}</p>
                        <div style={{marginTop: '15px'}}>
                           <span style={{fontSize: '0.85em', color: 'var(--accent-color)'}}>
                               Modo actual: {gameMode === 'pvai' ? 'Jugador vs IA' : 'Jugador vs Jugador'}
                           </span>
                        </div>
                    </div>

                    <div className="controls-box glass-card">
                        <h3>Controles</h3>
                        <div className="action-buttons">
                            <button onClick={toggleGameMode} className="action-btn secondary-btn">
                                {gameMode === 'pvai' ? <Users size={18} /> : <Bot size={18} />}
                                {gameMode === 'pvai' ? ' Cambiar a JvsJ' : ' Cambiar a JvsIA'}
                            </button>
                            <button onClick={resetGame} className="action-btn">
                                <RotateCcw size={18} /> Nueva Partida
                            </button>
                            <button onClick={undoMove} disabled={gameHistory.length < 2 || isThinking || (gameMode==='pvp' && gameHistory.length < 1)} className="action-btn">
                                <Undo2 size={18} /> Deshacer
                            </button>
                            <button onClick={() => setIsTimeModalOpen(true)} className="action-btn secondary-btn">
                                <Clock size={18} /> Control de Tiempo
                            </button>
                            <button onClick={onOpenSettings} className="action-btn secondary-btn">
                                <Settings size={18} /> Ajustes
                            </button>
                            <button onClick={handleCopyPGN} className="action-btn secondary-btn">
                                <Copy size={18} /> Copiar PGN
                            </button>
                        </div>
                    </div>
                </div>

            </div>
            <TimeModal 
                isOpen={isTimeModalOpen}
                onClose={() => setIsTimeModalOpen(false)}
                timeControl={timeControl}
                setTimeControl={setTimeControl}
                increment={increment}
                setIncrement={setIncrement}
                whiteTime={whiteTime}
                setWhiteTime={setWhiteTime}
                blackTime={blackTime}
                setBlackTime={setBlackTime}
                resetGame={resetGame}
            />
        </section>
    );
}
