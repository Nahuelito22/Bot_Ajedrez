import { useState } from 'react';
import { Settings } from 'lucide-react';

export default function Navbar({ activeSection, setActiveSection, onOpenSettings }) {
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
                        href="#proyecto" 
                        className={`nav-link ${activeSection === 'proyecto' ? 'active' : ''}`}
                        onClick={(e) => { e.preventDefault(); handleNavClick('proyecto'); }}
                    >
                        Sobre el Proyecto
                    </a>
                </li>
                <li>
                    <button 
                        onClick={onOpenSettings} 
                        className="action-btn" 
                        style={{ padding: '8px', marginLeft: '10px', background: 'transparent', boxShadow: 'none', color: 'var(--text-color)' }}
                        title="Configuración"
                    >
                        <Settings size={20} />
                    </button>
                </li>
            </ul>
        </nav>
    );
}
