import React, { useState } from 'react';

function Header({ user, onLogout }) {
    const [menuOpen, setMenuOpen] = useState(false);

    if (!user) {
        return <div className="header header--no-user">Utilisateur non connecté</div>;
    }

    const getInitial = (email) => email.charAt(0).toUpperCase();

    return (
        <div className="header">
            <h1 className="header__title">Agent Conversationnel</h1>
            <div className="header__user-menu">
                {/* Bouton utilisateur */}
                <button
                    className="header__user-menu__user-avatar"
                    onClick={() => setMenuOpen(!menuOpen)}
                >
                    {getInitial(user.email)}
                </button>

                {/* Menu déroulant */}
                {menuOpen && (
                    <div className="header__user-menu__dropdown">
                        <p className="header__user-menu__user-email">{user.email}</p>
                        <button
                            className="header__user-menu__dropdown__logout-button"
                            onClick={onLogout}
                        >
                            Déconnexion
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Header;
