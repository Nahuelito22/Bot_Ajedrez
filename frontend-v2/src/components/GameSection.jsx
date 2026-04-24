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

    const formatTime = (seconds) => {
        if (timeControl === 'unlimited') return "--:--";
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    const handleCopyPGN = () => {
        const pgn = getPGN();
        navigator.clipboard.writeText(pgn);
        alert('Partida (PGN) copiada al portapapeles!');
    };

    const toggleGameMode = () => {
        const newMode = gameMode === 'pvai' ? 'pvp' : 'pvai';
        setGameMode(newMode);
        resetGame();
    };

    const lightSquareStyle = boardTheme ? { backgroundColor: boardTheme.light } : { backgroundColor: '#f0d9b5' };
    const darkSquareStyle = boardTheme ? { backgroundColor: boardTheme.dark } : { backgroundColor: '#b58863' };

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
                
                {/* COLUMNA 1: JUGADORES */}
                <div className="players-column">
                    {/* SECCIÓN SUPERIOR (IA / JUGADOR 2) */}
                    <div className="player-section top-player" style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
                        <div className="player-info opponent-info glass-card" style={{ width: '250px', height: '163px' }}>
                            <div className="player-details">
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                    {gameMode === 'pvai' ? <Bot size={20} color="#cfb168"/> : <User size={20} color="#cfb168"/>}
                                    <h3 style={{ margin: 0 }}>{gameMode === 'pvai' ? 'Roque Chess AI' : 'Jugador 2 (Negras)'}</h3>
                                </div>
                                <p style={{ textAlign: 'center', margin: '5px 0 0 0', fontSize: '0.9em', opacity: 0.8 }}>
                                    {gameMode === 'pvai' ? 'Evaluación Posicional Avanzada' : 'Humano'}
                                </p>
                                {isThinking && <div className="thinking-indicator" style={{ marginTop: '10px' }}>Pensando...</div>}
                            </div>
                        </div>
                        
                        <div className={`clock-display ${isTimerRunning && (fen.split(' ')[1] === 'b') ? 'active-clock' : ''}`} style={{ width: '252px', height: '50px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            {formatTime(blackTime)}
                        </div>
                    </div>

                    {/* SECCIÓN INFERIOR (TÚ / JUGADOR 1) */}
                    <div className="player-section bottom-player" style={{ display: 'flex', flexDirection: 'column', gap: '19px' }}>
                        <div className={`clock-display ${isTimerRunning && (fen.split(' ')[1] === 'w') ? 'active-clock' : ''}`} style={{ width: '250px', height: '52px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            {formatTime(whiteTime)}
                        </div>

                        <div className="player-info user-info glass-card" style={{ width: '250px', height: '150px' }}>
                            <div className="player-details">
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                    <User size={20} color="#cfb168"/>
                                    <h3 style={{ margin: 0 }}>Jugador 1 (Blancas)</h3>
                                </div>
                                <p style={{ textAlign: 'center', margin: '5px 0 0 0', fontSize: '0.9em', opacity: 0.8 }}>Tú</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* COLUMNA 2: TABLERO */}
                <div className="board-column">
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
                </div>

                {/* COLUMNA 3: ESTADO Y CONTROLES */}
                <div className="controls-column">
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
                        <div className="action-buttons grid-buttons">
                            <button onClick={toggleGameMode} className="action-btn secondary-btn" style={{ gridColumn: '1 / -1' }}>
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
                                <Clock size={18} /> Tiempos
                            </button>
                            <button onClick={onOpenSettings} className="action-btn secondary-btn">
                                <Settings size={18} /> Ajustes
                            </button>
                        </div>
                    </div>
                </div>

                {/* COLUMNA 4: HISTORIAL */}
                <div className="history-column">
                    <div className="history-box glass-card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                            <h3 style={{ margin: 0 }}>Historial</h3>
                            <button onClick={handleCopyPGN} className="action-btn secondary-btn" style={{ padding: '6px 12px', fontSize: '0.8em' }}>
                                <Copy size={14} /> PGN
                            </button>
                        </div>
                        <div id="pgn" className="pgn-container" style={{ flexGrow: 1, overflowY: 'auto', fontFamily: 'monospace', fontSize: '0.9em', lineHeight: '1.5', paddingRight: '5px' }}>
                            {getPGN() || 'Aún no hay movimientos.'}
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
