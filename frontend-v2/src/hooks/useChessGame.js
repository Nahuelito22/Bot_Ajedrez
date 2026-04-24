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
            // Reemplaza esta URL por la correcta si es diferente
            const API_URL = "https://nahuelito22-bot-ajedrez.hf.space/predict_move"; 
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fen: game.fen() })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data && data.best_move) {
                makeMove(data.best_move);
            } else {
                console.error("No se recibió 'best_move' del bot:", data);
            }
        } catch (error) {
            console.error("Error al obtener jugada del bot:", error);
            setStatus("Error de conexión con el bot.");
        } finally {
            setIsThinking(false);
        }
    }, [game, makeMove]);

    // Disparar movimiento de IA cuando sea el turno de las negras
    useEffect(() => {
        if (game.turn() === 'b' && !game.isGameOver() && !isThinking) {
            const timeoutId = setTimeout(() => {
                getAiMove();
            }, 500); // Pequeño retraso para que se vea más natural
            return () => clearTimeout(timeoutId);
        }
    }, [game, isThinking, getAiMove]);

    const onDrop = (sourceSquare, targetSquare) => {
        if (game.turn() === 'b' || isThinking) return false;

        const move = {
            from: sourceSquare,
            to: targetSquare,
            promotion: 'q', // siempre promociona a reina por ahora, se puede mejorar
        };

        return makeMove(move);
    };

    const resetGame = () => {
        const newGame = new Chess();
        setGame(newGame);
        updateStatus(newGame);
    };

    const undoMove = () => {
        if (gameHistory.length < 2) return;
        const gameCopy = new Chess(game.fen());
        gameCopy.undo(); // Deshacer el de la IA
        gameCopy.undo(); // Deshacer el del jugador
        setGame(gameCopy);
        updateStatus(gameCopy);
    };

    return {
        fen,
        isThinking,
        status,
        gameHistory,
        onDrop,
        resetGame,
        undoMove,
        isGameOver: game.isGameOver()
    };
}
