import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const chatApi = {
    sendMessage: async (message, sessionId) => {
        const response = await axios.post(`${API_URL}/chat/chat`, {
            message,
            session_id: sessionId
        });
        return response.data;
    },

    sendRagMessage: async (message, sessionId) => {
        const response = await axios.post(`${API_URL}/chat/chat`, {
            message,
            session_id: sessionId,
            use_rag: true
        });
        return response.data;
    },

    sendRagUpload: async (files) => {
        const response = await axios.post(`${API_URL}/chat/uploadv2`, files);
        return response.data;
    },

    sendTeacherMessage: async (message, sessionId) => {
        let teacher_id;
        switch (sessionId) {
            case 'Maths':
                teacher_id = 'maths_teacher';
                break;
            case 'Histoire':
                teacher_id = 'histoire_teacher';
                break;
            case 'Français':
                teacher_id = 'francais_teacher';
                break;
            case 'RAG':
                teacher_id = 'rag_teacher';
                break;
            default:
                teacher_id = 'default_teacher';
        }
        const response = await axios.post(`${API_URL}/teacher/${teacher_id}/chat`, {
            teacher_id,
            message,
            session_id: sessionId
        });
        return response.data;
    },

    getHistorySession: async (sessionId) => {
        const response = await axios.get(`${API_URL}/chat/history/${sessionId}`);
        return response.data;
    },

    getHistoryTeacher: async (sessionId) => {
        const response = await axios.get(`${API_URL}/teacher/history/${sessionId}`);
        return response.data;
    },

    getAllSessions: async () => {
        const response = await axios.get(`${API_URL}/chat/sessions`);
        return response.data;
    }
};
