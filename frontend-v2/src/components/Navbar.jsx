import { useState } from 'react';
import { Settings } from 'lucide-react';

export default function Navbar({ activeSection, setActiveSection, appTheme, setAppTheme }) {
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const handleNavClick = (section) => {
        setActiveSection(section);
        setIsMenuOpen(false);
    };

    return (
        <nav className="navbar">
            <div className="nav-brand"><span className="brand-accent">Roque</span> Chess</div>
            <button 
                className={`hamburger-menu ${isMenuOpen ? 'active' : ''}`}
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                aria-label="Toggle menu"
            >
                <span className="hamburger-bar"></span>
                <span className="hamburger-bar"></span>
                <span className="hamburger-bar"></span>
            </button>
            <ul className={`nav-links ${isMenuOpen ? 'active' : ''}`}>
                <li>
                    <a 
                        href="#jugar" 
                        className={`nav-link ${activeSection === 'jugar' ? 'active' : ''}`}
                        onClick={(e) => { e.preventDefault(); handleNavClick('jugar'); }}
                    >
                        Jugar
                    </a>
                </li>
                <li>
                    <a 
                        href="#analisis" 
                        className={`nav-link ${activeSection === 'analisis' ? 'active' : ''}`}
                        onClick={(e) => { e.preventDefault(); handleNavClick('analisis'); }}
                    >
                        Análisis del Modelo
                    </a>
                </li>
                <li>
                    <a 
                        href="#ai_analisis" 
                        className={`nav-link ${activeSection === 'ai_analisis' ? 'active' : ''}`}
                        onClick={(e) => { e.preventDefault(); handleNavClick('ai_analisis'); }}
                    >
                        Análisis de IA
                    </a>
                </li>
                <li>
                    <a 
                        href="#proyecto" 
                        className={`nav-link ${activeSection === 'proyecto' ? 'active' : ''}`}
                        onClick={(e) => { e.preventDefault(); handleNavClick('proyecto'); }}
                    >
                        Sobre el Proyecto
                    </a>
                </li>
                <li>
                    <a 
                        href="https://github.com/Nahuelito22/Bot_Ajedrez" 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="nav-link"
                    >
                        Código Fuente
                    </a>
                </li>
                <li>
                    <div className="theme-switch-wrapper" style={{display: 'flex', alignItems: 'center', marginLeft: '10px'}}>
                        <label className="theme-switch" htmlFor="checkbox">
                            <input 
                                type="checkbox" 
                                id="checkbox" 
                                checked={appTheme === 'light'} 
                                onChange={() => setAppTheme(appTheme === 'dark' ? 'light' : 'dark')}
                            />
                            <div className="slider round">
                                <div className="theme-icon">
                                    <img src="/chesspieces/wikipedia/wP.png" alt="Theme Icon" style={{width: '20px'}}/>
                                </div>
                            </div>
                        </label>
                    </div>
                </li>
            </ul>
        </nav>
    );
}
