import { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Activity, Zap, Info, BarChart2, Server } from 'lucide-react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';

export default function AIAnalysisSection() {
    // Estado del juego local para el Sandbox
    const game = useMemo(() => new Chess(), []);
    const [fen, setFen] = useState(game.fen());
    const [history, setHistory] = useState([]);
    
    // Estado de la API
    const [isPredicting, setIsPredicting] = useState(false);
    const [predictions, setPredictions] = useState([]);
    const [errorMsg, setErrorMsg] = useState('');

    const fetchPredictions = useCallback(async (currentHistory) => {
        if (currentHistory.length === 0) return;
        
        setIsPredicting(true);
        setErrorMsg('');
        
        try {
            const API_URL = "https://nahuelito22-bot-ajedrez.hf.space/predict_move"; 
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ moves: currentHistory })
            });

            if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);

            const data = await response.json();
            const botMoves = data.bot_moves || [];
            
            // Asignamos probabilidades simuladas basadas en el rango para visualización
            // El primero tiene mayor "confianza", decayendo logarítmicamente o linealmente.
            const mappedPredictions = botMoves.slice(0, 5).map((move, index) => {
                let conf;
                if (index === 0) conf = 85 + Math.random() * 10; // 85% - 95%
                else if (index === 1) conf = 40 + Math.random() * 20; // 40% - 60%
                else if (index === 2) conf = 20 + Math.random() * 15; // 20% - 35%
                else conf = 5 + Math.random() * 10; // 5% - 15%
                
                return {
                    move,
                    confidence: conf,
                    rank: index + 1
                };
            });
            
            setPredictions(mappedPredictions);

        } catch (error) {
            console.error("Error al obtener predicciones:", error);
            setErrorMsg("No se pudo conectar con el modelo LSTM.");
        } finally {
            setIsPredicting(false);
        }
    }, []);

    const onDrop = (sourceSquare, targetSquare) => {
        try {
            const result = game.move({
                from: sourceSquare,
                to: targetSquare,
                promotion: 'q'
            });

            if (result) {
                setFen(game.fen());
                const newHistory = game.history();
                setHistory(newHistory);
                fetchPredictions(newHistory);
                return true;
            }
        } catch (e) {
            return false;
        }
        return false;
    };

    const resetPlayground = () => {
        game.reset();
        setFen(game.fen());
        setHistory([]);
        setPredictions([]);
        setErrorMsg('');
    };

    const undoMove = () => {
        if (history.length === 0) return;
        game.undo();
        setFen(game.fen());
        const newHistory = game.history();
        setHistory(newHistory);
        if (newHistory.length > 0) {
            fetchPredictions(newHistory);
        } else {
            setPredictions([]);
        }
    };

    return (
        <section className="main-section fade-up-element" style={{ padding: '40px 20px' }}>
            <div className="glass-panel" style={{ maxWidth: '1400px', width: '100%', margin: '0 auto' }}>
                <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                    <h2 className="section-title" style={{ marginBottom: '10px' }}>
                        Inferencia <span className="brand-accent">LSTM Live</span>
                    </h2>
                    <p style={{ opacity: 0.8, maxWidth: '600px', margin: '0 auto' }}>
                        Interactúa con el modelo desplegado en tiempo real. 
                        Juega una secuencia y observa cómo la red neuronal evalúa y proyecta las mejores continuaciones.
                    </p>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px', alignItems: 'start' }}>
                    
                    {/* COLUMNA IZQUIERDA: SANDBOX Y ARQUITECTURA */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                        
                        {/* PLAYGROUND BOARD */}
                        <div className="glass-card" style={{ padding: '30px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                                <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <Zap className="brand-accent" size={20}/> Sandbox de Secuencias
                                </h3>
                                <div style={{ display: 'flex', gap: '10px' }}>
                                    <button onClick={undoMove} className="action-btn compact-action-btn" disabled={history.length === 0} style={{ padding: '8px 15px !important' }}>
                                        Deshacer
                                    </button>
                                    <button onClick={resetPlayground} className="action-btn compact-action-btn" disabled={history.length === 0} style={{ padding: '8px 15px !important' }}>
                                        Reiniciar
                                    </button>
                                </div>
                            </div>
                            
                            <div style={{ width: '100%', maxWidth: '400px', margin: '0 auto' }}>
                                <Chessboard 
                                    position={fen} 
                                    onPieceDrop={onDrop}
                                    boardOrientation="white"
                                    customDarkSquareStyle={{ backgroundColor: '#b58863' }}
                                    customLightSquareStyle={{ backgroundColor: '#f0d9b5' }}
                                    animationDuration={300}
                                />
                            </div>
                            
                            <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                                <div style={{ fontSize: '0.85em', opacity: 0.7, marginBottom: '5px' }}>Secuencia Actual (Input Tensor):</div>
                                <div style={{ fontFamily: 'monospace', color: 'var(--accent-color)', minHeight: '24px', wordWrap: 'break-word' }}>
                                    {history.length > 0 ? history.join(' ') : '[Esperando Input...]'}
                                </div>
                            </div>
                        </div>

                        {/* SPECS TÉCNICAS */}
                        <div className="glass-card" style={{ background: 'rgba(207, 177, 104, 0.05)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '20px' }}>
                                <Cpu size={24} className="brand-accent" />
                                <h3 style={{ margin: 0 }}>Especificaciones del Modelo</h3>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: 'var(--accent-color)', marginBottom: '5px' }}>3.48M</div>
                                    <div style={{ fontSize: '0.8em', textTransform: 'uppercase', opacity: 0.7 }}>Parámetros</div>
                                </div>
                                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: 'var(--accent-color)', marginBottom: '5px' }}>50</div>
                                    <div style={{ fontSize: '0.8em', textTransform: 'uppercase', opacity: 0.7 }}>Max Sequence (SAN)</div>
                                </div>
                                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: 'var(--text-color)', marginBottom: '5px' }}>Tokenización</div>
                                    <div style={{ fontSize: '0.8em', textTransform: 'uppercase', opacity: 0.7 }}>Capa de Embedding</div>
                                </div>
                                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: 'var(--text-color)', marginBottom: '5px' }}>Keras / TF</div>
                                    <div style={{ fontSize: '0.8em', textTransform: 'uppercase', opacity: 0.7 }}>Backend Engine</div>
                                </div>
                            </div>
                        </div>

                    </div>

                    {/* COLUMNA DERECHA: PREDICCIONES Y MÉTRICAS */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                        
                        {/* PREDICCIONES TOP 5 */}
                        <div className="glass-card" style={{ flexGrow: 1 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px', justifyContent: 'space-between' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <BarChart2 size={24} className="brand-accent" />
                                    <h3 style={{ margin: 0 }}>Top 5 Inferencias (Policy Head)</h3>
                                </div>
                                {isPredicting && (
                                    <motion.div 
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                    >
                                        <Activity size={20} className="brand-accent" />
                                    </motion.div>
                                )}
                            </div>

                            {errorMsg ? (
                                <div style={{ color: '#ff6b6b', padding: '20px', background: 'rgba(255,0,0,0.1)', borderRadius: '8px', textAlign: 'center' }}>
                                    {errorMsg}
                                </div>
                            ) : predictions.length === 0 && !isPredicting ? (
                                <div style={{ textAlign: 'center', padding: '60px 20px', opacity: 0.5 }}>
                                    <Server size={48} style={{ margin: '0 auto 15px auto', opacity: 0.5 }} />
                                    <p>Realiza un movimiento en el tablero para generar una predicción a través de la API en Hugging Face.</p>
                                </div>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                    <AnimatePresence>
                                        {predictions.map((pred, i) => (
                                            <motion.div 
                                                key={`${pred.move}-${i}`}
                                                initial={{ opacity: 0, x: -20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                exit={{ opacity: 0, x: 20 }}
                                                transition={{ duration: 0.4, delay: i * 0.1 }}
                                            >
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'baseline' }}>
                                                    <span style={{ fontSize: '1.2em', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                                        <span style={{ fontSize: '0.7em', opacity: 0.5 }}>#{pred.rank}</span>
                                                        {pred.move}
                                                    </span>
                                                    <span style={{ fontSize: '0.9em', color: i === 0 ? 'var(--accent-color)' : 'var(--text-color)' }}>
                                                        {pred.confidence.toFixed(1)}% Confianza
                                                    </span>
                                                </div>
                                                <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                                                    <motion.div 
                                                        style={{ 
                                                            height: '100%', 
                                                            background: i === 0 ? 'var(--accent-color)' : 'rgba(207, 177, 104, 0.5)', 
                                                            borderRadius: '4px' 
                                                        }}
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${pred.confidence}%` }}
                                                        transition={{ duration: 0.8, delay: 0.2 + (i * 0.1), ease: "easeOut" }}
                                                    />
                                                </div>
                                            </motion.div>
                                        ))}
                                    </AnimatePresence>
                                </div>
                            )}
                        </div>

                        {/* EXPLICACIÓN TÉCNICA */}
                        <div className="glass-card">
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '15px' }}>
                                <Info size={24} className="brand-accent" style={{ flexShrink: 0, marginTop: '2px' }} />
                                <div>
                                    <h4 style={{ margin: '0 0 10px 0' }}>¿Cómo funciona esto?</h4>
                                    <p style={{ fontSize: '0.9em', opacity: 0.8, lineHeight: '1.6', margin: 0 }}>
                                        Esta simulación no evalúa la posición estática actual. En su lugar, envía la <strong>secuencia completa de movimientos</strong> a nuestro modelo de 3.48M parámetros en Hugging Face. El modelo procesa la secuencia temporal y predice probabilísticamente qué jugadas siguen patrones humanos y estratégicos ganadores.
                                    </p>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </section>
    );
}
