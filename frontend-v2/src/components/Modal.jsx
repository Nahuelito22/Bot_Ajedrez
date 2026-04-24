import { X } from 'lucide-react';

export default function Modal({ isOpen, onClose, title, imgSrc, description }) {
    if (!isOpen) return null;

    return (
        <div className="modal" style={{ display: 'block' }} onClick={(e) => {
            if (e.target.className === 'modal') onClose();
        }}>
            <div className="modal-content glass-card fade-in">
                <span className="close-modal" onClick={onClose}>
                    <X size={32} />
                </span>
                <h2>{title}</h2>
                <div className="modal-body">
                    <img src={imgSrc} alt="Gráfico Expandido" />
                    <p>{description}</p>
                </div>
            </div>
        </div>
    );
}
