import React from 'react';
import ConversationsList from './ConversationsList';
import AgentList from './AgentList';

function Sidebar({ agents, sessions, currentSession, onSessionChange }) {
    return (
        <div className="sidebar">
            {/* Liste des agents */}
            <div className="sidebar__section sidebar__section">
                <AgentList 
                    agents={agents}
                    currentSession={currentSession}
                    onSessionChange={onSessionChange}
                />
            </div>
            {/* Liste des conversations */}
            <div className="sidebar__section">
                <ConversationsList
                    sessions={sessions}
                    currentSession={currentSession}
                    onSessionChange={onSessionChange}
                />
            </div>
        </div>
    );
}

export default Sidebar;
