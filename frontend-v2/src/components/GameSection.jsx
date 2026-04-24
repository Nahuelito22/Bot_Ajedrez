import { useState, useRef, useEffect } from 'react';
import { Chessboard } from 'react-chessboard';
import { RotateCcw, Undo2, Settings, Clock, Copy, User, Bot, Users } from 'lucide-react';
import { Rnd } from 'react-rnd';
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

    const [positions, setPositions] = useState({
        player2: { x: 10, y: 10, width: 250, height: 120 },
        player1: { x: 10, y: 550, width: 250, height: 120 },
        status: { x: 850, y: 10, width: 300, height: 150 },
        controls: { x: 850, y: 180, width: 300, height: 250 },
        history: { x: 1180, y: 10, width: 280, height: 400 }
    });

    const handleCopyLayout = () => {
        const layoutStr = JSON.stringify(positions, null, 2);
        navigator.clipboard.writeText(layoutStr);
        alert('¡Coordenadas del diseño copiadas al portapapeles!\nPuedes pegarlas en el chat.');
    };

    return (
        <section id="jugar" className="main-section" style={{ position: 'relative', minHeight: '800px', width: '100%' }}>
            
            {/* Botón Flotante para Copiar Layout */}
            <div style={{ position: 'fixed', bottom: '20px', right: '20px', zIndex: 3000 }}>
                <button onClick={handleCopyLayout} className="action-btn" style={{ boxShadow: '0 0 15px rgba(207, 177, 104, 0.6)' }}>
                    <Copy size={18} style={{ marginRight: '5px' }}/> Copiar Layout
                </button>
            </div>

            {/* TABLERO CENTRADO ESTATICO */}
            <div style={{ position: 'absolute', top: '50px', left: '300px', width: '500px' }} ref={boardWrapperRef}>
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

            {/* JUGADOR 2 */}
            <Rnd
                bounds="parent"
                size={{ width: positions.player2.width, height: positions.player2.height }}
                position={{ x: positions.player2.x, y: positions.player2.y }}
                onDragStop={(e, d) => setPositions(p => ({ ...p, player2: { ...p.player2, x: d.x, y: d.y } }))}
                onResizeStop={(e, direction, ref, delta, position) => {
                    setPositions(p => ({ ...p, player2: { width: ref.style.width, height: ref.style.height, ...position } }));
                }}
            >
                <div className="player-info opponent-info glass-card" style={{ width: '100%', height: '100%', margin: 0, boxSizing: 'border-box' }}>
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
            </Rnd>

            {/* JUGADOR 1 */}
            <Rnd
                bounds="parent"
                size={{ width: positions.player1.width, height: positions.player1.height }}
                position={{ x: positions.player1.x, y: positions.player1.y }}
                onDragStop={(e, d) => setPositions(p => ({ ...p, player1: { ...p.player1, x: d.x, y: d.y } }))}
                onResizeStop={(e, direction, ref, delta, position) => {
                    setPositions(p => ({ ...p, player1: { width: ref.style.width, height: ref.style.height, ...position } }));
                }}
            >
                <div className="player-info user-info glass-card" style={{ width: '100%', height: '100%', margin: 0, boxSizing: 'border-box' }}>
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
            </Rnd>

            {/* ESTADO */}
            <Rnd
                bounds="parent"
                size={{ width: positions.status.width, height: positions.status.height }}
                position={{ x: positions.status.x, y: positions.status.y }}
                onDragStop={(e, d) => setPositions(p => ({ ...p, status: { ...p.status, x: d.x, y: d.y } }))}
                onResizeStop={(e, direction, ref, delta, position) => {
                    setPositions(p => ({ ...p, status: { width: ref.style.width, height: ref.style.height, ...position } }));
                }}
            >
                <div className="status-box glass-card" style={{ width: '100%', height: '100%', margin: 0, boxSizing: 'border-box' }}>
                    <h3>Estado de la Partida</h3>
                    <p id="gameStatus" className="status-text">{status}</p>
                    <div style={{marginTop: '15px'}}>
                        <span style={{fontSize: '0.85em', color: 'var(--accent-color)'}}>
                            Modo actual: {gameMode === 'pvai' ? 'Jugador vs IA' : 'Jugador vs Jugador'}
                        </span>
                    </div>
                </div>
            </Rnd>

            {/* CONTROLES */}
            <Rnd
                bounds="parent"
                size={{ width: positions.controls.width, height: positions.controls.height }}
                position={{ x: positions.controls.x, y: positions.controls.y }}
                onDragStop={(e, d) => setPositions(p => ({ ...p, controls: { ...p.controls, x: d.x, y: d.y } }))}
                onResizeStop={(e, direction, ref, delta, position) => {
                    setPositions(p => ({ ...p, controls: { width: ref.style.width, height: ref.style.height, ...position } }));
                }}
            >
                <div className="controls-box glass-card" style={{ width: '100%', height: '100%', margin: 0, boxSizing: 'border-box' }}>
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
                    </div>
                </div>
            </Rnd>

            {/* HISTORIAL PGN */}
            <Rnd
                bounds="parent"
                size={{ width: positions.history.width, height: positions.history.height }}
                position={{ x: positions.history.x, y: positions.history.y }}
                onDragStop={(e, d) => setPositions(p => ({ ...p, history: { ...p.history, x: d.x, y: d.y } }))}
                onResizeStop={(e, direction, ref, delta, position) => {
                    setPositions(p => ({ ...p, history: { width: ref.style.width, height: ref.style.height, ...position } }));
                }}
            >
                <div className="history-box glass-card" style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', margin: 0, boxSizing: 'border-box' }}>
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
            </Rnd>

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
