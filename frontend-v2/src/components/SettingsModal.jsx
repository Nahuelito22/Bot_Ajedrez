import { X } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose, boardTheme, setBoardTheme, appTheme, setAppTheme }) {
    if (!isOpen) return null;

    const boardThemes = [
        { id: 'default', name: 'Clásico', light: '#f0d9b5', dark: '#b58863' },
        { id: 'grayscale', name: 'Gris', light: '#e1e1e1', dark: '#aaaaaa' },
        { id: 'blue', name: 'Azul', light: '#dee3e6', dark: '#8ca2ad' },
        { id: 'green', name: 'Verde', light: '#ffffdd', dark: '#86a666' }
    ];

    return (
        <div className="modal" style={{ display: 'block' }} onClick={(e) => {
            if (e.target.className === 'modal') onClose();
        }}>
            <div className="modal-content glass-card fade-in">
                <span className="close-modal" onClick={onClose}>
                    <X size={28} />
                </span>
                <h2 style={{marginTop: 0, borderBottom: '1px solid var(--border-color)', paddingBottom: '10px'}}>Personalización</h2>
                
                <div className="settings-section" style={{marginTop: '20px'}}>
                    <h3>Tema de la Aplicación</h3>
                    <div className="options" style={{display: 'flex', gap: '10px'}}>
                        <button 
                            className={`action-btn ${appTheme === 'dark' ? 'selected' : ''}`}
                            onClick={() => setAppTheme('dark')}
                            style={{ opacity: appTheme === 'dark' ? 1 : 0.6 }}
                        >
                            Modo Oscuro
                        </button>
                        <button 
                            className={`action-btn ${appTheme === 'light' ? 'selected' : ''}`}
                            onClick={() => setAppTheme('light')}
                            style={{ opacity: appTheme === 'light' ? 1 : 0.6 }}
                        >
                            Modo Claro
                        </button>
                    </div>
                </div>

                <div className="settings-section" style={{marginTop: '20px'}}>
                    <h3>Color del Tablero</h3>
                    <div className="options" style={{display: 'flex', gap: '10px', flexWrap: 'wrap'}}>
                        {boardThemes.map(theme => (
                            <div 
                                key={theme.id}
                                onClick={() => setBoardTheme(theme)}
                                style={{
                                    cursor: 'pointer',
                                    padding: '10px',
                                    borderRadius: '8px',
                                    border: `2px solid ${boardTheme.id === theme.id ? 'var(--accent-color)' : 'transparent'}`,
                                    display: 'flex',
                                    flexDirection: 'column',
                                    alignItems: 'center',
                                    gap: '5px'
                                }}
                            >
                                <div style={{ display: 'flex', width: '40px', height: '40px', borderRadius: '4px', overflow: 'hidden' }}>
                                    <div style={{ flex: 1, backgroundColor: theme.light }}></div>
                                    <div style={{ flex: 1, backgroundColor: theme.dark }}></div>
                                </div>
                                <span style={{fontSize: '0.8em'}}>{theme.name}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Futuro: Controles de tiempo podrían ir aquí si los implementas en useChessGame */}
            </div>
        </div>
    );
}
