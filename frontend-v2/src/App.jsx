import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import GameSection from './components/GameSection';
import AnalysisSection from './components/AnalysisSection';
import AboutSection from './components/AboutSection';
import Modal from './components/Modal';
import SettingsModal from './components/SettingsModal';

function App() {
    const [activeSection, setActiveSection] = useState('jugar');
    const [modalData, setModalData] = useState({ isOpen: false, imgSrc: '', title: '', description: '' });
    
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [appTheme, setAppTheme] = useState(localStorage.getItem('theme') || 'dark');
    const [boardTheme, setBoardTheme] = useState({ id: 'default', name: 'Clásico', light: '#f0d9b5', dark: '#b58863' });
    const [pieceTheme, setPieceTheme] = useState('wikipedia');

    useEffect(() => {
        if (appTheme === 'light') {
            document.body.classList.add('light-theme');
            localStorage.setItem('theme', 'light');
        } else {
            document.body.classList.remove('light-theme');
            localStorage.setItem('theme', 'dark');
        }
    }, [appTheme]);

    const openModal = (imgSrc, title, description) => {
        setModalData({ isOpen: true, imgSrc, title, description });
    };

    const closeModal = () => {
        setModalData(prev => ({ ...prev, isOpen: false }));
    };

    return (
        <>
            <Navbar 
                activeSection={activeSection} 
                setActiveSection={setActiveSection} 
                appTheme={appTheme}
                setAppTheme={setAppTheme}
            />
            
            <main className="content-container">
                {activeSection === 'jugar' && <GameSection boardTheme={boardTheme} pieceTheme={pieceTheme} onOpenSettings={() => setIsSettingsOpen(true)} />}
                {activeSection === 'analisis' && <AnalysisSection onOpenModal={openModal} />}
                {activeSection === 'proyecto' && <AboutSection />}
            </main>

            <Modal 
                isOpen={modalData.isOpen}
                onClose={closeModal}
                title={modalData.title}
                imgSrc={modalData.imgSrc}
                description={modalData.description}
            />

            <SettingsModal
                isOpen={isSettingsOpen}
                onClose={() => setIsSettingsOpen(false)}
                boardTheme={boardTheme}
                setBoardTheme={setBoardTheme}
                pieceTheme={pieceTheme}
                setPieceTheme={setPieceTheme}
                appTheme={appTheme}
                setAppTheme={setAppTheme}
            />
        </>
    );
}

export default App;
