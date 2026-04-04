// src/pages/HomePage.jsx
import React from 'react';
import { Link } from 'react-router-dom';

function HomePage() {
    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-50">
            <h1 className="text-3xl font-bold mb-4">Bienvenue sur l'agent conversationnel</h1>
            <Link to="/chat" className="px-4 py-2 bg-blue-500 text-white rounded">
                Commencer une conversation
            </Link>
        </div>
    );
}

export default HomePage;
