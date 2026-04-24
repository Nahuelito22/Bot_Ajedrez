export default function AboutSection() {
    return (
        <section id="proyecto" className="main-section">
             <div className="about-container glass-panel">
                <h2 className="section-title">Sobre este Proyecto</h2>
                <div className="about-grid">
                    <div className="about-card glass-card fade-up-element">
                        <h3>Inspiración</h3>
                        <p>Este proyecto nace de la fascinación por el ajedrez y la inteligencia artificial. A diferencia de los motores tradicionales que se basan en el cálculo bruto, el objetivo era crear un bot que jugara con un estilo "humano", imitando la intuición y el reconocimiento de patrones de un jugador experimentado.</p>
                    </div>
                    
                    <div className="about-card glass-card fade-up-element" style={{animationDelay: '0.1s'}}>
                        <h3>El Desafío Técnico</h3>
                        <p>El mayor reto fue transformar la representación visual del tablero en datos procesables por la red neuronal, utilizando técnicas de Natural Language Processing (NLP) como Word2Vec para entender las secuencias de movimientos como si fueran "oraciones" en un idioma.</p>
                    </div>
                    
                    <div className="about-card glass-card fade-up-element" style={{animationDelay: '0.2s'}}>
                        <h3>Entrega Final Coderhouse</h3>
                        <p>Desarrollado como proyecto final para la carrera de Data Science, obteniendo la calificación máxima. Representa la culminación de meses de estudio en machine learning, deep learning y procesamiento de datos.</p>
                    </div>
                </div>
            </div>
        </section>
    );
}
