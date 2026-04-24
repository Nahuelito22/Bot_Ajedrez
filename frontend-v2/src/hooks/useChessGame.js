import { useState, useCallback, useEffect } from 'react';
import { Chess } from 'chess.js';

export function useChessGame() {
    const [game, setGame] = useState(new Chess());
    const [fen, setFen] = useState(game.fen());
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
        setFen(currentGame.fen());
        setGameHistory(currentGame.history());
    }, []);

    const makeMove = useCallback((move) => {
        const gameCopy = new Chess(game.fen());
        try {
            const result = gameCopy.move(move);
            if (result) {
                setGame(gameCopy);
                updateStatus(gameCopy);
                return true;
            }
        } catch (e) {
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
            const timeoutId = setTimeout(() => getAiMove(), 500);
            return () => clearTimeout(timeoutId);
        }
    }, [game, isThinking, getAiMove]);

    // Lógica para Drag & Drop y Click-to-Move
    const [moveFrom, setMoveFrom] = useState('');
    const [optionSquares, setOptionSquares] = useState({});

    function getMoveOptions(square) {
        const moves = game.moves({ square, verbose: true });
        if (moves.length === 0) {
            setOptionSquares({});
            return false;
        }

        const newSquares = {};
        moves.map((move) => {
            newSquares[move.to] = {
                background: game.get(move.to) && game.get(move.to).color !== game.get(square).color
                    ? 'radial-gradient(circle, rgba(0,0,0,.1) 85%, transparent 85%)'
                    : 'radial-gradient(circle, rgba(0,0,0,.1) 25%, transparent 25%)',
                borderRadius: '50%'
            };
            return move;
        });
        newSquares[square] = { background: 'rgba(255, 255, 0, 0.4)' };
        setOptionSquares(newSquares);
        return true;
    }

    const onSquareClick = (square) => {
        // No permitir hacer click si el bot piensa o le toca al bot
        if (game.turn() === 'b' || isThinking) return;

        // Limpiar estilos anteriores
        setOptionSquares({});

        // Si ya seleccionamos una pieza de origen
        if (moveFrom) {
            const move = {
                from: moveFrom,
                to: square,
                promotion: 'q' // Siempre promocionar a reina en esta versión
            };

            const isValidMove = makeMove(move);
            
            if (!isValidMove) {
                // Si el movimiento falló pero hicimos click en otra de nuestras piezas, seleccionarla
                if (game.get(square) && game.get(square).color === game.turn()) {
                    setMoveFrom(square);
                    getMoveOptions(square);
                    return;
                }
                setMoveFrom('');
                setOptionSquares({});
                return;
            }

            setMoveFrom('');
            setOptionSquares({});
            return;
        }

        // Si no hemos seleccionado origen, intentar seleccionar pieza
        if (game.get(square) && game.get(square).color === game.turn()) {
            setMoveFrom(square);
            getMoveOptions(square);
        }
    };

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

    const resetGame = () => {
        const newGame = new Chess();
        setGame(newGame);
        setMoveFrom('');
        setOptionSquares({});
        updateStatus(newGame);
    };

    const undoMove = () => {
        if (gameHistory.length < 2) return;
        const gameCopy = new Chess(game.fen());
        gameCopy.undo(); 
        gameCopy.undo(); 
        setGame(gameCopy);
        setMoveFrom('');
        setOptionSquares({});
        updateStatus(gameCopy);
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
        isGameOver: game.isGameOver()
    };
}
