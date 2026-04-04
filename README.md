# Projet TP MIAGE M2 - Agent Conversationnel

## Structure du Projet

```
C:.
├───api/                    # Gestion des routes et endpoints de l'API
│   ├───endpoints/         # Endpoints spécifiques par fonctionnalité
│   │   └───chat.py       # Endpoint pour les fonctionnalités de chat
│   └───router.py         # Router principal regroupant tous les endpoints
├───core/                  # Configuration et éléments centraux de l'application
├───models/               # Modèles de données Pydantic
│   └───chat.py          # Modèles pour les requêtes/réponses de chat
├───services/            # Services métier
│   └───llm_service.py   # Service d'interaction avec le LLM
├───tests/            # Test unitaire sur le code
│   └───test_mongo_service.py   # test lié à la base de donnée
├───utils/               # Utilitaires et helpers
└───main.py             # Point d'entrée de l'application
```


## Réalisation
- Un projet fonctionnel avec les endpoints du tp1 du tp2
- Des endpoints supplémentaires pour gérer les conversations dans la BD MongoDB (effacer des conversations et lister toutes les sessions)
- Des enpoints supplémentaires (2 endpoints) pour le projet chatbot educatif (seulement des prototypes, le endpoint cours (/ask) ne fonctionne pas)
- Le projet est lié au backend MongoDB avec une seule collection pour le moment qui est conversation
- Le projet est lié au front React mais aucun changement n'a été apporté au repo de base
- Le dossier de tests a été mis en place avec un premier test unitaire

## Lancement du projet et des test 

### le projet backend/FASTAPI

option1: 
via le debugger python fastapi

option2:
```bash
cd app
uvicorn main:app --reload
```

### le projet front
```bash
npm start
```

### les tests
```bash
cd app
pytest
```

## Installation et Configuration

### Prérequis
- Python 3.11+ (ici : https://www.python.org/downloads/release/python-3110/ il faut redémarrer après installation pour avoir le $PATH sur l'OS)
- Visual Studio Code avec l'extension Python
- Une clé OpenAI que je vais vous fournir

### Installation

1. **Cloner le projet**
```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_PROJET>
```

2. **Créer l'environnement virtuel**
```bash
python -m venv venv
```

3. **Activer l'environnement virtuel**
- Windows :
```bash
.\venv\Scripts\activate
```
- macOS/Linux :
```bash
source venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

5. **Configurer la clé API OpenAI**
Créer un fichier `.env` à la racine du projet :
```
OPENAI_API_KEY=votre-clé-api-openai
```


## Explication des Composants

### 1. Main Application (`main.py`)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
```
- Point d'entrée de l'application
- Configure FastAPI et les middlewares
- Initialise les routes

### 2. Modèles (`models/chat.py`)
```python
class ChatRequest(BaseModel):
    message: str
```
- Définit la structure des données entrantes/sortantes
- Utilise Pydantic pour la validation des données
- Version simple pour débuter, extensible pour le contexte

### 3. Service LLM (`services/llm_service.py`)
```python
class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(...)
```
- Gère l'interaction avec le modèle de langage
- Configure le client OpenAI
- Traite les messages et le contexte

### 4. Router API (`api/router.py`)
```python
@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
```
- Définit les endpoints de l'API
- Gère les requêtes HTTP
- Valide les données entrantes

## Utilisation de l'API

### Version Simple
```bash
curl -X 'POST' \
  'http://localhost:8000/chat/simple' \
  -H 'Content-Type: application/json' \
  -d '{"message": "Bonjour!"}'
```

### Version avec Contexte
```bash
curl -X 'POST' \
  'http://localhost:8000/chat/with-context' \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Bonjour!",
    "context": [
      {"role": "user", "content": "Comment vas-tu?"},
      {"role": "assistant", "content": "Je vais bien, merci!"}
    ]
  }'
```

## Debugging avec VS Code

1. Ouvrir le projet dans VS Code
2. Aller dans la section "Run and Debug" (Ctrl + Shift + D)
3. Sélectionner la configuration "Python: FastAPI"
4. Appuyer sur F5 ou cliquer sur le bouton Play
5. Démarrer Swagger : http://127.0.0.1:8000/docs

## Structure de l'API

### Endpoints Disponibles

- `/chat/simple` : Version basique sans contexte
- `/chat/with-context` : Version avancée avec gestion du contexte

### Flux de Données

1. La requête arrive sur l'endpoint
2. Les modèles Pydantic valident les données
3. Le service LLM traite la demande
4. La réponse est formatée et renvoyée

## Progression Pédagogique

1. **Démarrer avec la version simple**
   - Comprendre la structure de base
   - Tester les appels API simples

2. **Évoluer vers la version avec contexte**
   - Ajouter la gestion de l'historique
   - Comprendre l'importance du contexte dans les LLM

3. **Explorer les fonctionnalités avancées**
   - Implémenter des prompts personnalisés
   - Gérer différents types de réponses

## Dépannage

### Problèmes Courants

1. **Erreur de clé API**
   - Vérifier le fichier `.env`
   - S'assurer que la clé est valide

2. **Erreurs de dépendances**
   - Vérifier l'activation du venv
   - Réinstaller les requirements

3. **Erreurs de contexte**
   - Vérifier le format du contexte
   - S'assurer que les rôles sont valides

4. **Powershell**
   - Si les droits admin ne sont pas présent : ''Set-ExecutionPolicy Unrestricted -Scope CurrentUser -Force''

## Ressources

- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation LangChain](https://python.langchain.com/)
- [API OpenAI](https://platform.openai.com/docs/api-reference)

# **Agent Conversationnel - Frontend**

Ce projet est l'interface frontend de notre agent conversationnel éducatif. Il utilise React et SCSS pour offrir une expérience utilisateur moderne et personnalisée.

## **Structure et Architecture**

Le projet est organisé de manière modulaire pour faciliter la maintenance et l'évolutivité.

### **Arborescence du Projet**

```plaintext
src/
├── components/         # Composants réutilisables
│   ├── AgentList.jsx
│   ├── ChatWindow.jsx
│   ├── ConversationsList.jsx
│   ├── Header.jsx
│   ├── Message.jsx
│   ├── MessageInput.jsx
│   └── Sidebar.jsx
├── pages/              # Pages principales
│   ├── ChatPage.jsx
│   └── HomePage.jsx
├── services/           # Services pour les appels API
│   └── api.js
├── styles/             # Fichiers SCSS organisés par composant/page
│   ├── AgentList.scss
│   ├── ChatPage.scss
│   ├── ChatWindow.scss
│   ├── ConversationsList.scss
│   ├── Header.scss
│   ├── MessageInput.scss
│   ├── Sidebar.scss
│   ├── settings.scss   # Variables SCSS globales
│   ├── variables.scss  # Variables spécifiques (ex : couleurs, breakpoints)
│   └── index.scss      # Import centralisé des styles
├── App.js              # Point d'entrée principal de l'application
├── index.js            # Point d'entrée React
└── clippyjs.d.ts       # Types pour la bibliothèque ClippyJS
```

---

## **Description des Composants**

### **1. AgentList**

- **Rôle** : Affiche une liste d'agents conversationnels disponibles (ex. Maths, Français, Histoire).
- **Style associé** : `AgentList.scss`

### **2. sdfChatWindows**

- **Rôle** : Affiche l'historique des messages sous forme de conversation.
- **Fonctionnalité** : Inclut un défilement fluide automatique au chargement et à l'ajout de nouveaux messages.
- **Style associé** : `ChatWindow.scss`

### **3. ConversationsList**

- **Rôle** : Liste des conversations enregistrées.
- **Fonctionnalité** : Permet de sélectionner ou de créer une nouvelle session de conversation.
- **Style associé** : `ConversationsList.scss`

### **4. Header**

- **Rôle** : Barre supérieure de navigation affichant le nom de l'utilisateur et un menu déroulant.
- **Fonctionnalité** : Inclut un bouton de déconnexion.
- **Style associé** : `Header.scss`

### **5. ChatWindow**

- **Rôle** : Affiche un message individuel (utilisateur ou assistant) avec un formatage markdown si nécessaire.
- **Style associé** : Aucun spécifique (intégré à `ChatWindow.scss`).

### **6. MessageInput**

- **Rôle** : Champ de saisie pour envoyer un message.
- **Fonctionnalité** : Désactive l'envoi lorsque le message est vide ou que l'envoi est en cours.
- **Style associé** : `MessageInput.scss`

### **7. Sidebar**

- **Rôle** : Barre latérale regroupant les composants `ConversationsList` et `AgentList`.
- **Style associé** : `Sidebar.scss`

---

## **Pages**

### **1. ChatPage**

- **Description** : Page principale où l'utilisateur interagit avec l'agent conversationnel.
- **Structure** :
  - `Header` (haut de la page)
  - `Sidebar` (à gauche)
  - `ChatWindow` et `MessageInput` (au centre)
- **Style associé** : `ChatPage.scss`

### **2. HomePage**

- **Description** : Page d'accueil du projet (), servant de point d'entrée, n'a pas de fonctionnalités particulières.
- **Fonctionnalité** : Contient un lien pour accéder à la `ChatPage`.

---

## **Architecture SCSS**

- Chaque composant ou page a son propre fichier SCSS.
- Les fichiers SCSS incluent uniquement les styles nécessaires au composant ou à la page concernée.
- Les variables globales (couleurs, breakpoints, etc.) sont définies dans `variables.scss` et importées dans chaque fichier.
- `index.scss` centralise l'import de tous les styles.

### **Variables principales (dans **variable.scss**)**

- **Couleurs** :
  ```scss
  $primary-color: #145da0;
  $secondary-color: #0c2d48;
  $accent-color: #2e8bc0;
  $background-color: #b1d4e0;
  ```

---

## **Initialisation du Projet**

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/Xolitor/chatbot-frontend.git
   cd chatbot-frontend
   ```

2. **Installer les dépendances :**

   ```bash
   npm install
   ```

3. **Lancer le serveur backend :**

   - Assurez-vous que le backend est configuré et démarré (instructions spécifiques au backend).

4. **Lancer le serveur frontend :**

   ```bash
   npm start
   ```

   - L'application sera accessible sur `http://localhost:3000`.

---

## **Démarrage**

- Ouvrez `http://localhost:3000` dans votre navigateur.
- Utilisez l'interface pour démarrer une nouvelle conversation ou accéder à une session existante.
