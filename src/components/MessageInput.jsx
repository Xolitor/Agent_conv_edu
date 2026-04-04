import React, { useState, useEffect } from 'react';
import clippyIcon from '../clippy.png';
import { chatApi } from '../services/api';

function MessageInput({currentSession, onSendMessage, isLoading }) {
    const [inputValue, setInputValue] = useState(''); // État pour gérer l'entrée utilisateur
    const [isNowLoading, setIsNowLoading] = useState(isLoading); // État pour gérer le chargement

    const handleChange = (e) => {
        setInputValue(e.target.value); // Met à jour l'état avec la valeur du champ d'entrée
    };

    useEffect(() => {
        setIsNowLoading(isLoading);
    }, [isLoading]);

    const handleSendMessage = () => {
        if (inputValue.trim() === '') return; // Ne pas envoyer de message vide
        onSendMessage(inputValue); // Appeler la fonction parent avec le message
        setInputValue(''); // Réinitialiser le champ d'entrée
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    };

    const handleFileUpload = () => {
        // Affiche le message de chargement
        // Création d'un élément input de type "file"
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.pdf, .html, text/html, application/pdf'; // Filtre les fichiers acceptés
        input.multiple = true; // Autorise un seul fichier (modifier si besoin)
    
        // Écoute de l'événement de changement (l'utilisateur a sélectionné un fichier)
        input.onchange = (e) => {
            const files = e.target.files; // Récupère les fichiers sélectionnés
            const allowedTypes = ['application/pdf', 'text/html'];
            const validFiles = Array.from(files).filter(file => allowedTypes.includes(file.type));
            
            if (validFiles.length > 0) {
                handleUploadToDB(validFiles); // Appelle la fonction pour uploader vers la DB
            } else {
                alert('Veuillez sélectionner uniquement des fichiers PDF ou HTML.');
            }
        };
    
        // Déclenche la boîte de dialogue de sélection de fichier
        input.click();
    };

    const handleUploadToDB = async (files) => {
        setIsNowLoading(true);
        try {
            const formData = new FormData();
            files.forEach((file, index) => {
                formData.append(`files`, file); // Ajoute chaque fichier à FormData
            });

            const response = await chatApi.sendRagUpload(formData, currentSession);
        } catch (error) {
            setIsNowLoading(false);
            console.error('Error sending message:', error);
        } finally {
            setIsNowLoading(false);
        }
    }

    return (
        <div className="message-input">
            <div className="input-field">
                {currentSession === "RAG" && (
                    <button onClick={handleFileUpload} className="upload-button">
                        <img src={clippyIcon} alt="Upload"  />
                    </button>
                )}
                <input
                    type="text"
                    placeholder="Tapez votre message ici..."
                    value={inputValue}
                    onChange={handleChange}
                    onKeyPress={handleKeyPress}
                    disabled={isNowLoading}
                />
            </div>
            <button
                className="send-button"
                onClick={handleSendMessage}
                disabled={isNowLoading || inputValue.trim() === ''}
            >
                {isNowLoading ? 'Envoi...' : 'Envoyer'}
            </button>
        </div>
    );
}

export default MessageInput;
