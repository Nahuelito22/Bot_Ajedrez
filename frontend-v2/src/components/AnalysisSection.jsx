import { Expand } from 'lucide-react';

export default function AnalysisSection({ onOpenModal }) {
    return (
        <section id="analisis" className="main-section">
            <div className="analysis-container glass-panel">
                <h2 className="section-title">Análisis del Modelo</h2>
                
                <div className="analysis-intro fade-up-element" style={{marginBottom: '2rem', color: 'var(--text-color)', opacity: 0.9, lineHeight: 1.6}}>
                    <p>El desempeño de <strong>Roque Chess</strong> ha sido validado mediante un riguroso análisis cuantitativo. El modelo, basado en una arquitectura de Red Neuronal Recurrente (LSTM), fue entrenado con un dataset de <strong>1 millón de partidas</strong> de jugadores de alto nivel (2000+ ELO).</p>
                    <p>Para evaluar su fuerza real, se organizó un <strong>Torneo Gauntlet de 700 partidas</strong> contra diferentes niveles del motor Stockfish. Los siguientes gráficos detallan cómo el bot aplica su "intuición" aprendida en comparación con el cálculo algorítmico tradicional.</p>
                </div>

                <div className="dashboard-grid">
                    <div 
                        className="chart-container glass-card clickable-card fade-up-element" 
                        onClick={() => onOpenModal('/img/grafico_de_rendimiento.png', 'Rendimiento Promedio', 'Evolución del desempeño del bot en las diferentes etapas de la partida (Apertura, Medio Juego, Final).')}
                    >
                        <div className="chart-header">
                            <h3>Rendimiento Promedio</h3>
                        </div>
                        <div className="chart-wrapper">
                            <img src="/img/grafico_de_rendimiento.png" alt="Gráfico de Rendimiento Promedio" />
                            <div className="chart-overlay">
                                <Expand size={24} />
                                <span>Ver más</span>
                            </div>
                        </div>
                        <p className="chart-description">Evolución del desempeño del bot en las diferentes etapas de la partida.</p>
                    </div>
                    
                    <div 
                        className="chart-container glass-card clickable-card fade-up-element" 
                        style={{animationDelay: '0.1s'}} 
                        onClick={() => onOpenModal('/img/grafico_de_errores.png', 'Calidad Táctica', 'Análisis de la pérdida de centipeones. Refleja la precisión del modelo en comparación con los motores tradicionales.')}
                    >
                        <div className="chart-header">
                            <h3>Calidad Táctica</h3>
                        </div>
                        <div className="chart-wrapper">
                            <img src="/img/grafico_de_errores.png" alt="Gráfico de Pérdida de Centipeones" />
                            <div className="chart-overlay">
                                <Expand size={24} />
                                <span>Ver más</span>
                            </div>
                        </div>
                        <p className="chart-description">Análisis de la precisión del modelo respecto a centipeones perdidos.</p>
                    </div>
                </div>
            </div>
        </section>
    );
}
