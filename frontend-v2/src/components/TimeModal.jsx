import { X } from 'lucide-react';

export default function TimeModal({ 
    isOpen, 
    onClose, 
    timeControl, 
    setTimeControl, 
    increment, 
    setIncrement, 
    resetGame 
}) {
    if (!isOpen) return null;

    const handleApply = () => {
        // Al aplicar un nuevo control de tiempo, reiniciamos la partida
        resetGame();
        onClose();
    };

    return (
        <div className="modal" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'var(--modal-backdrop)', position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: 2000 }} onClick={(e) => {
            if (e.target.className === 'modal') onClose();
        }}>
            <div className="modal-content glass-card fade-in" style={{position: 'relative', width: '90%', maxWidth: '400px'}}>
                <span className="close-modal" onClick={onClose} style={{position: 'absolute', top: '15px', right: '15px', cursor: 'pointer', color: 'var(--text-color)'}}>
                    <X size={28} />
                </span>
                <h2 style={{marginTop: 0, borderBottom: '1px solid var(--glass-border)', paddingBottom: '10px'}}>Control de Tiempo</h2>
                
                <div className="settings-section" style={{marginTop: '20px'}}>
                    <div style={{marginBottom: '15px'}}>
                        <label style={{display: 'block', marginBottom: '8px', color: 'var(--text-color)'}}>Tiempo Inicial:</label>
                        <select 
                            value={timeControl} 
                            onChange={(e) => setTimeControl(e.target.value)}
                            style={{
                                width: '100%',
                                padding: '10px',
                                borderRadius: '8px',
                                background: 'var(--primary-bg)',
                                color: 'var(--text-color)',
                                border: '1px solid var(--glass-border)',
                                outline: 'none',
                                fontSize: '1rem',
                                cursor: 'pointer'
                            }}
                        >
                            <option value="unlimited">Ilimitado</option>
                            <option value="60">1 minuto</option>
                            <option value="180">3 minutos</option>
                            <option value="300">5 minutos</option>
                            <option value="600">10 minutos</option>
                            <option value="1800">30 minutos</option>
                        </select>
                    </div>

                    <div style={{marginBottom: '20px'}}>
                        <label style={{display: 'block', marginBottom: '8px', color: 'var(--text-color)'}}>Incremento por jugada:</label>
                        <select 
                            value={increment} 
                            onChange={(e) => setIncrement(parseInt(e.target.value))}
                            disabled={timeControl === 'unlimited'}
                            style={{
                                width: '100%',
                                padding: '10px',
                                borderRadius: '8px',
                                background: 'var(--primary-bg)',
                                color: 'var(--text-color)',
                                border: '1px solid var(--glass-border)',
                                outline: 'none',
                                fontSize: '1rem',
                                cursor: timeControl === 'unlimited' ? 'not-allowed' : 'pointer',
                                opacity: timeControl === 'unlimited' ? 0.5 : 1
                            }}
                        >
                            <option value="0">0 segundos</option>
                            <option value="1">1 segundo</option>
                            <option value="2">2 segundos</option>
                            <option value="5">5 segundos</option>
                            <option value="10">10 segundos</option>
                        </select>
                    </div>

                    <div style={{display: 'flex', justifyContent: 'flex-end', gap: '10px'}}>
                        <button onClick={onClose} className="action-btn secondary-btn" style={{padding: '8px 15px'}}>
                            Cancelar
                        </button>
                        <button onClick={handleApply} className="action-btn" style={{padding: '8px 15px'}}>
                            Aplicar
                        </button>
                    </div>
                    <p style={{fontSize: '0.85em', color: 'var(--accent-color)', marginTop: '15px', textAlign: 'center'}}>
                        Nota: Aplicar un nuevo control de tiempo reiniciará la partida actual.
                    </p>
                </div>
            </div>
        </div>
    );
}
