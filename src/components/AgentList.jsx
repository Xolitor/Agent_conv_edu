import React from 'react';

const AgentList = ({ agents, currentSession, onSessionChange }) => {
    return (
        <div className="agent__list">
            <h2 className="sidebar__title">Agents</h2>
                <ul className="sidebar__list">
                    {agents
                        .filter((agent) => agent.trim() !== "")
                        .map((agent) => (
                    <li key={agent} 
                        onClick={() => onSessionChange(agent)}
                        className={`agent-list__item ${
                                currentSession === agent
                                    ? 'agent-list__item--active'
                                    : ''
                            }`}>{agent}</li>
                    ))}
                    {/* <li className="sidebar__item sidebar__item--agent">Français</li>
                    <li className="sidebar__item sidebar__item--agent">Histoire</li> */}
                </ul>
        </div>
    );
};

export default AgentList;