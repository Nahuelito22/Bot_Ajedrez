import { Chessboard } from 'react-chessboard';
import { RotateCcw, Undo2 } from 'lucide-react';
import { useChessGame } from '../hooks/useChessGame';

export default function GameSection({ boardTheme }) {
    const {
        fen,
        isThinking,
        status,
        gameHistory,
        onDrop,
        onSquareClick,
        optionSquares,
        resetGame,
        undoMove
    } = useChessGame();

    // Colores por defecto si no hay tema seleccionado
    const lightSquareStyle = boardTheme ? { backgroundColor: boardTheme.light } : { backgroundColor: '#f0d9b5' };
    const darkSquareStyle = boardTheme ? { backgroundColor: boardTheme.dark } : { backgroundColor: '#b58863' };

    return (
        <section id="jugar" className="main-section">
            <div className="game-container glass-panel fade-up-element">
                
                <div className="left-panel">
                    <div className="player-info bot-info glass-card">
                        <div className="avatar bot-avatar"></div>
                        <div className="player-details">
                            <h3>Roque Chess AI</h3>
                            <p>Evaluación Posicional Avanzada</p>
                            {isThinking && <div className="thinking-indicator">Pensando...</div>}
                        </div>
                    </div>

                    <div className="board-wrapper">
                        <Chessboard 
                            id="BasicBoard" 
                            position={fen} 
                            onPieceDrop={onDrop}
                            onSquareClick={onSquareClick}
                            customSquareStyles={optionSquares}
                            boardOrientation="white"
                            customDarkSquareStyle={darkSquareStyle}
                            customLightSquareStyle={lightSquareStyle}
                            animationDuration={200}
                        />
                    </div>

                    <div className="player-info user-info glass-card">
                        <div className="avatar user-avatar"></div>
                        <div className="player-details">
                            <h3>Jugador</h3>
                            <p>Tú</p>
                        </div>
                    </div>
                </div>

                <div className="right-panel">
                    <div className="status-box glass-card">
                        <h3>Estado de la Partida</h3>
                        <p id="gameStatus" className="status-text">{status}</p>
                    </div>

                    <div className="controls-box glass-card">
                        <h3>Controles</h3>
                        <div className="action-buttons">
                            <button onClick={resetGame} className="action-btn">
                                <RotateCcw size={18} style={{marginRight: '8px', verticalAlign: 'middle'}}/>
                                Nueva Partida
                            </button>
                            <button onClick={undoMove} disabled={gameHistory.length < 2 || isThinking} className="action-btn">
                                <Undo2 size={18} style={{marginRight: '8px', verticalAlign: 'middle'}}/>
                                Deshacer
                            </button>
                        </div>
                    </div>
                </div>

            </div>
        </section>
    );
}
