import { useState, useCallback, useEffect, useMemo } from 'react';
import { Chess } from 'chess.js';

export function useChessGame() {
    // Instancia única y persistente del motor de ajedrez
    const game = useMemo(() => new Chess(), []);
    
    // Estados para la UI
    const [fen, setFen] = useState(game.fen());
    const [isThinking, setIsThinking] = useState(false);
    const [status, setStatus] = useState('Turno de las blancas');
    const [gameHistory, setGameHistory] = useState([]);
    const [optionSquares, setOptionSquares] = useState({});
    const [moveFrom, setMoveFrom] = useState('');

    const [gameMode, setGameMode] = useState('pvai'); // 'pvai' | 'pvp'
    const [timeControl, setTimeControl] = useState('unlimited'); // 'unlimited' | 'custom'
    const [whiteTime, setWhiteTime] = useState(600); // en segundos
    const [blackTime, setBlackTime] = useState(600);
    const [increment, setIncrement] = useState(0);
    const [isTimerRunning, setIsTimerRunning] = useState(false);

    const updateUI = useCallback(() => {
        setFen(game.fen());
        setGameHistory(game.history());
        
        let statusText = '';
        let moveColor = game.turn() === 'b' ? 'negras' : 'blancas';

        if (game.isCheckmate()) {
            statusText = `Juego terminado, ganan las ${moveColor === 'blancas' ? 'negras' : 'blancas'} por Jaque Mate.`;
        } else if (game.isDraw()) {
            statusText = 'Juego terminado en tablas.';
        } else {
            statusText = `Turno de las ${moveColor}`;
            if (game.isCheck()) {
                statusText += `, las ${moveColor} están en Jaque.`;
            }
        }
        
        if (whiteTime === 0 && timeControl !== 'unlimited') {
            statusText = 'Juego terminado, ganan las negras por tiempo.';
        } else if (blackTime === 0 && timeControl !== 'unlimited') {
            statusText = 'Juego terminado, ganan las blancas por tiempo.';
        }

        setStatus(statusText);
    }, [game, whiteTime, blackTime, timeControl]);

    const makeMove = useCallback((move) => {
        try {
            const currentTurn = game.turn();
            const result = game.move(move);
            if (result) {
                if (timeControl !== 'unlimited' && increment > 0 && !game.isGameOver()) {
                    if (currentTurn === 'w') {
                        setWhiteTime(prev => prev + increment);
                    } else {
                        setBlackTime(prev => prev + increment);
                    }
                }
                if (!isTimerRunning) setIsTimerRunning(true);
                updateUI();
                return true;
            }
        } catch (e) {
            console.warn("Movimiento inválido detectado:", e.message);
        }
        return false;
    }, [game, updateUI, timeControl, increment, isTimerRunning]);

    const getAiMove = useCallback(async () => {
        if (game.isGameOver() || game.turn() === 'w') return;

        setIsThinking(true);
        setStatus("El bot está pensando...");

        try {
            const API_URL = "https://nahuelito22-bot-ajedrez.hf.space/predict_move"; 
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ moves: game.history() })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            const botMoves = data.bot_moves || [];
            let moveMade = false;

            for (const move of botMoves) {
                if (makeMove(move)) {
                    moveMade = true;
                    break;
                }
            }

            if (!moveMade) {
                const possibleMoves = game.moves();
                if (possibleMoves.length > 0) {
                    const randomIdx = Math.floor(Math.random() * possibleMoves.length);
                    makeMove(possibleMoves[randomIdx]);
                }
            }
        } catch (error) {
            console.error("Error al obtener jugada del bot:", error);
            setStatus("Error de conexión con el bot.");
        } finally {
            setIsThinking(false);
        }
    }, [game, makeMove]);

    useEffect(() => {
        if (gameMode === 'pvai' && game.turn() === 'b' && !game.isGameOver() && !isThinking) {
            const timer = setTimeout(getAiMove, 500);
            return () => clearTimeout(timer);
        }
    }, [game, fen, isThinking, getAiMove, gameMode]);

    useEffect(() => {
        let interval;
        if (isTimerRunning && timeControl !== 'unlimited' && !game.isGameOver() && whiteTime > 0 && blackTime > 0) {
            interval = setInterval(() => {
                const currentTurn = game.turn();
                if (currentTurn === 'w') {
                    setWhiteTime((prev) => {
                        if (prev <= 1) {
                            setIsTimerRunning(false);
                            return 0;
                        }
                        return prev - 1;
                    });
                } else {
                    setBlackTime((prev) => {
                        if (prev <= 1) {
                            setIsTimerRunning(false);
                            return 0;
                        }
                        return prev - 1;
                    });
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isTimerRunning, timeControl, fen, whiteTime, blackTime]);

    // Sincronizar relojes cuando cambia el control de tiempo
    useEffect(() => {
        const startTime = timeControl === 'unlimited' ? 600 : parseInt(timeControl);
        setWhiteTime(startTime);
        setBlackTime(startTime);
        setIsTimerRunning(false);
    }, [timeControl]);

    const onDrop = (sourceSquare, targetSquare) => {
        if ((gameMode === 'pvai' && game.turn() === 'b') || isThinking || whiteTime === 0 || blackTime === 0) return false;

        const result = makeMove({
            from: sourceSquare,
            to: targetSquare,
            promotion: 'q'
        });

        if (result) {
            setMoveFrom('');
            setOptionSquares({});
            return true;
        }
        return false;
    };

    const onSquareClick = (square) => {
        if ((gameMode === 'pvai' && game.turn() === 'b') || isThinking || whiteTime === 0 || blackTime === 0) return;

        // Intentar mover si ya hay una pieza seleccionada
        if (moveFrom) {
            const result = makeMove({ from: moveFrom, to: square, promotion: 'q' });
            if (result) {
                setMoveFrom('');
                setOptionSquares({});
                return;
            }
        }

        // Selección de pieza y visualización de opciones
        const moves = game.moves({ square, verbose: true });
        if (moves.length > 0) {
            setMoveFrom(square);
            const newSquares = {};
            moves.forEach((m) => {
                newSquares[m.to] = {
                    background: game.get(m.to) ? 'radial-gradient(circle, rgba(0,0,0,.1) 85%, transparent 85%)' : 'radial-gradient(circle, rgba(0,0,0,.1) 25%, transparent 25%)',
                    borderRadius: '50%'
                };
            });
            newSquares[square] = { background: 'rgba(255, 255, 0, 0.4)' };
            setOptionSquares(newSquares);
        } else {
            setMoveFrom('');
            setOptionSquares({});
        }
    };

    const resetGame = () => {
        game.reset();
        setIsTimerRunning(false);
        const startTime = timeControl === 'unlimited' ? 600 : parseInt(timeControl);
        setWhiteTime(startTime);
        setBlackTime(startTime);
        updateUI();
        setMoveFrom('');
        setOptionSquares({});
    };

    const undoMove = () => {
        if (game.history().length < 2) return;
        game.undo();
        if (gameMode === 'pvai') game.undo(); // Undo 2 if against AI
        updateUI();
        setMoveFrom('');
        setOptionSquares({});
    };

    const getPGN = () => {
        return game.pgn();
    };

    return {
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
        isGameOver: game.isGameOver() || (timeControl !== 'unlimited' && (whiteTime === 0 || blackTime === 0)),
        gameMode, setGameMode,
        timeControl, setTimeControl,
        whiteTime, setWhiteTime,
        blackTime, setBlackTime,
        increment, setIncrement,
        isTimerRunning, setIsTimerRunning
    };
}
