import React, { useState, useEffect } from 'react';
import Sidebar from '../components/Sidebar';
import Header from '../components/Header';
import ChatWindow from '../components/ChatWindow';
import MessageInput from '../components/MessageInput';
import { chatApi } from '../services/api';
import { v4 as uuidv4 } from 'uuid';

// import clippy from 'clippyjs/dist/clippy';

// const clippy = require('clippyjs');
 
// clippy.load('Merlin', (agent) => {
//     // do anything with the loaded agent
//     agent.show();
// });

function ChatPage({ user, onLogout }) {
    const [messages, setMessages] = useState([]);
    const [sessions, setSessions] = useState([]);
    const [agents, setAgents] = useState(['Maths', 'Français', 'Histoire', 'RAG']);
    const [currentSession, setCurrentSession] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const newSession = `session-${uuidv4()}`;
        setCurrentSession(newSession);
        loadSessions();
    }, []);

    useEffect(() => {
        if (currentSession && currentSession !== 'NewSession') {
            loadHistory();
        } else if (currentSession === 'NewSession') {
            const newSession = `session-${uuidv4()}`;
            setCurrentSession(newSession);
            loadSessions();
        }
    }, [currentSession]);

    const loadSessions = async () => {
        try {
            const response = await chatApi.getAllSessions();
            setSessions(response);
        } catch (error) {
            console.error('Error loading sessions:', error);
        }
    };

    const loadHistory = async () => {
        try {
            if (currentSession === 'RAG') {
                // Clear messages for RAG demonstration
                setMessages([]);
                return;
            }
            
            if (currentSession === 'Maths' || currentSession === 'Français' || currentSession === 'Histoire') {
                const history = await chatApi.getHistoryTeacher(currentSession);
                setMessages(history);
                return;
            } else{
                const history = await chatApi.getHistorySession(currentSession);
                setMessages(history);
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    };

    // clippy.load('Merlin', (agent) => {
    //     // do anything with the loaded agent
    //     agent.show();
    // });

    const handleSendMessage = async (content) => {
        if (!currentSession) return;
        setIsLoading(true);
    
        // Ajouter le message utilisateur à la conversation immédiatement
        setMessages(prev => [...prev, { role: 'user', content }]);
        
        try {
            let response;
            
            if (currentSession === 'Maths' || currentSession === 'Français' || currentSession === 'Histoire') {
                response = await chatApi.sendTeacherMessage(content, currentSession);
            } else if (currentSession === 'RAG') {
                response = await chatApi.sendRagMessage(content, currentSession);
            } else {
                response = await chatApi.sendMessage(content, currentSession);
            }
            
            // Vérifier si la réponse est dans un format valide
            if (!response || typeof response !== 'object') {
                throw new Error('Format de réponse invalide');
            }
            
            // Normaliser la réponse pour assurer un format cohérent
            const responseContent = typeof response.response === 'string' 
                ? response.response 
                : 'La réponse n\'a pas pu être formatée correctement';
            
            // Ajouter uniquement la réponse de l'assistant
            setMessages(prev => [
                ...prev,
                { 
                    role: 'assistant', 
                    content: responseContent,
                    metadata: response.metadata || null 
                }
            ]);
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => [
                ...prev,
                { 
                    role: 'assistant',
                    content: `Désolé, une erreur s'est produite lors du traitement de votre demande. (${error.message || 'Erreur inconnue'})` 
                }
            ]);
        } finally {
            loadSessions();
            setIsLoading(false);
        }
    };

    return (
        <div className="chat-page">
            <Header user={user} onLogout={onLogout} />
            <div className="chat-page__content">
                <Sidebar
                    agents={agents}
                    sessions={sessions}
                    currentSession={currentSession}
                    onSessionChange={setCurrentSession}
                />
                <div className="chat-page__chat-container">
                    <ChatWindow messages={messages} />
                    <MessageInput currentSession={currentSession} onSendMessage={handleSendMessage} isLoading={isLoading} />
                </div>
            </div>
        </div>
    );
}

export default ChatPage;
