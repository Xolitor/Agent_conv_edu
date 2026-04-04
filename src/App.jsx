// src/App.jsx
import React from 'react';
import './styles/index.scss';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import HomePage from './pages/HomePage';

function App() {
    const user = { name: 'Utilisateur', email: 'user@example.com' };

    const handleLogout = () => {
        // Déconnexion logic here
        window.location.href = '/';
    };

    return (
        <Router>
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/chat" element={<ChatPage user={user} onLogout={handleLogout} />} />
            </Routes>
        </Router>
    );
}

export default App;
