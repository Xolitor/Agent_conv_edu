import { createRef, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Message from './Message';

const ChatWindow = ({ messages }) => {
    const chatWindowRef = createRef();

    useEffect(() => {
        if (chatWindowRef.current) {
            chatWindowRef.current.scrollTop = 0;

            chatWindowRef.current.scrollTo({
                top: chatWindowRef.current.scrollHeight,
                behavior: "smooth",
            });
        }
    }, []);

    useEffect(() => {
        if (chatWindowRef.current) {
            chatWindowRef.current.scrollTo({
                top: chatWindowRef.current.scrollHeight,
                behavior: "smooth",
            });
        }
    }, [messages]);

    return (
        <div className="chat-window" ref={chatWindowRef} >
            {messages.map((message, index) => (
                <div
                    key={index}
                    className={`message ${message.role}`}
                >
                    {message.role === 'assistant' ? (
                        <ReactMarkdown className="bubble">{message.content}</ReactMarkdown>
                    ) : (
                        <div className="bubble">{message.content}</div>
                    )}
                </div>
        ))}
</div>

    );
};

export default ChatWindow;
