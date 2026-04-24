import { useState, useCallback, useEffect } from 'react';
import { Chess } from 'chess.js';

export function useChessGame() {
    const [game, setGame] = useState(new Chess());
    const [isThinking, setIsThinking] = useState(false);
    const [status, setStatus] = useState('Turno de las blancas');
    const [gameHistory, setGameHistory] = useState([]);

    const updateStatus = useCallback((currentGame) => {
        let statusText = '';
        let moveColor = currentGame.turn() === 'b' ? 'negras' : 'blancas';

        if (currentGame.isCheckmate()) {
            statusText = `Juego terminado, ganan las ${moveColor === 'blancas' ? 'negras' : 'blancas'} por Jaque Mate.`;
        } else if (currentGame.isDraw()) {
            statusText = 'Juego terminado en tablas.';
        } else {
            statusText = `Turno de las ${moveColor}`;
            if (currentGame.isCheck()) {
                statusText += `, las ${moveColor} están en Jaque.`;
            }
        }
        setStatus(statusText);
        setGameHistory(currentGame.history());
    }, []);

    const makeMove = useCallback((move) => {
        try {
            const gameCopy = new Chess(game.fen());
            const result = gameCopy.move(move);
            if (result) {
                setGame(gameCopy);
                updateStatus(gameCopy);
                return true;
            }
        } catch (e) {
            console.warn("Movimiento inválido:", e.message);
            return false;
        }
        return false;
    }, [game, updateStatus]);

    const getAiMove = useCallback(async () => {
        if (game.isGameOver() || game.turn() === 'w') return;

        setIsThinking(true);
        setStatus("El bot está pensando...");

        try {
            const API_URL = "https://nahuelito22-bot-ajedrez.hf.space/predict_move"; 
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fen: game.fen() })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            if (data && data.best_move) {
                // El bot devuelve el movimiento en formato SAN o objeto, makeMove lo maneja
                makeMove(data.best_move);
            }
        } catch (error) {
            console.error("Error al obtener jugada del bot:", error);
            setStatus("Error de conexión con el bot.");
        } finally {
            setIsThinking(false);
        }
    }, [game, makeMove]);

    useEffect(() => {
        if (game.turn() === 'b' && !game.isGameOver() && !isThinking) {
            const timer = setTimeout(getAiMove, 500);
            return () => clearTimeout(timer);
        }
    }, [game, isThinking, getAiMove]);

    // Lógica de interacción
    const [moveFrom, setMoveFrom] = useState('');
    const [optionSquares, setOptionSquares] = useState({});

    const onDrop = (sourceSquare, targetSquare) => {
        if (game.turn() === 'b' || isThinking) return false;

        const move = {
            from: sourceSquare,
            to: targetSquare,
            promotion: 'q'
        };

        const result = makeMove(move);
        if (result) {
            setMoveFrom('');
            setOptionSquares({});
        }
        return result;
    };

    const onSquareClick = (square) => {
        if (game.turn() === 'b' || isThinking) return;

        // Si ya hay una pieza seleccionada e intentamos mover
        if (moveFrom) {
            const result = makeMove({ from: moveFrom, to: square, promotion: 'q' });
            if (result) {
                setMoveFrom('');
                setOptionSquares({});
                return;
            }
        }

        // Si no hay selección o el movimiento fue inválido, intentar seleccionar nueva pieza
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

    return {
        fen: game.fen(),
        isThinking,
        status,
        gameHistory,
        onDrop,
        onSquareClick,
        optionSquares,
        resetGame: () => {
            const newGame = new Chess();
            setGame(newGame);
            updateStatus(newGame);
            setMoveFrom('');
            setOptionSquares({});
        },
        undoMove: () => {
            if (gameHistory.length < 2) return;
            const g = new Chess(game.fen());
            g.undo(); g.undo();
            setGame(g);
            updateStatus(g);
        },
        isGameOver: game.isGameOver()
    };
}
