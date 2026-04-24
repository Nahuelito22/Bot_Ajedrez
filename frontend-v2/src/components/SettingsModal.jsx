import { X } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose, boardTheme, setBoardTheme, pieceTheme, setPieceTheme, appTheme, setAppTheme }) {
    if (!isOpen) return null;

    const boardThemes = [
        { id: 'default', name: 'Clásico', light: '#f0d9b5', dark: '#b58863' },
        { id: 'grayscale', name: 'Gris', light: '#e1e1e1', dark: '#aaaaaa' },
        { id: 'blue', name: 'Azul', light: '#dee3e6', dark: '#8ca2ad' },
        { id: 'green', name: 'Verde', light: '#ffffdd', dark: '#86a666' }
    ];

    const pieceThemesList = [
        "alpha", "anarcandy", "caliente", "california", "cardinal", "cburnett", "celtic", "chess7",
        "chessnut", "companion", "cooke", "dubrovny", "fantasy", "firi", "fresca",
        "gioco", "governor", "horsey", "icpieces", "kiwen-suwi", "kosal", "leipzig", "letter",
        "maestro", "merida", "monarchy", "mpchess", "pirouetti", "pixel", "reillycraig",
        "rhosgfx", "riohacha", "shapes", "spatial", "staunty", "tatiana", "uscf", "wikipedia", "xkcd"
    ];

    return (
        <div className="modal" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'var(--modal-backdrop)', position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: 2000 }} onClick={(e) => {
            if (e.target.className === 'modal') onClose();
        }}>
            <div className="modal-content glass-card fade-in" style={{position: 'relative', width: '90%', maxWidth: '500px', maxHeight: '90vh', overflowY: 'auto'}}>
                <span className="close-modal" onClick={onClose} style={{position: 'absolute', top: '15px', right: '15px', cursor: 'pointer', color: 'var(--text-color)'}}>
                    <X size={28} />
                </span>
                <h2 style={{marginTop: 0, borderBottom: '1px solid var(--glass-border)', paddingBottom: '10px'}}>Personalización</h2>
                
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
                                    gap: '5px',
                                    background: 'rgba(255,255,255,0.05)'
                                }}
                            >
                                <div style={{ display: 'flex', width: '40px', height: '40px', borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                                    <div style={{ flex: 1, backgroundColor: theme.light }}></div>
                                    <div style={{ flex: 1, backgroundColor: theme.dark }}></div>
                                </div>
                                <span style={{fontSize: '0.8em'}}>{theme.name}</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="settings-section" style={{marginTop: '20px'}}>
                    <h3>Estilo de Piezas</h3>
                    <div className="options">
                        <select 
                            value={pieceTheme} 
                            onChange={(e) => setPieceTheme(e.target.value)}
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
                            {pieceThemesList.map(theme => (
                                <option key={theme} value={theme}>
                                    {theme.charAt(0).toUpperCase() + theme.slice(1)}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div style={{marginTop: '15px', display: 'flex', justifyContent: 'center', gap: '10px'}}>
                        {['wP', 'wN', 'wB', 'wR', 'wQ', 'wK'].map(p => (
                            <img 
                                key={p} 
                                src={`/chesspieces/${pieceTheme}/${p}.${['alpha', 'uscf', 'wikipedia'].includes(pieceTheme) && pieceTheme !== 'wikipedia' ? 'svg' : pieceTheme === 'wikipedia' ? 'png' : 'svg'}`} 
                                alt={p} 
                                style={{width: '35px', height: '35px'}}
                            />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
