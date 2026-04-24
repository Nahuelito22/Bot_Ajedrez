import { useState } from 'react';
import Navbar from './components/Navbar';
import GameSection from './components/GameSection';
import AnalysisSection from './components/AnalysisSection';
import AboutSection from './components/AboutSection';
import Modal from './components/Modal';

function App() {
    const [activeSection, setActiveSection] = useState('jugar');
    const [modalData, setModalData] = useState({ isOpen: false, imgSrc: '', title: '', description: '' });

    const openModal = (imgSrc, title, description) => {
        setModalData({ isOpen: true, imgSrc, title, description });
    };

    const closeModal = () => {
        setModalData(prev => ({ ...prev, isOpen: false }));
    };

    return (
        <>
            <Navbar activeSection={activeSection} setActiveSection={setActiveSection} />
            
            <main className="content-container">
                {activeSection === 'jugar' && <GameSection />}
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
        </>
    );
}

export default App;
