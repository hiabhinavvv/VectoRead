import React, { useEffect, useRef, useState } from 'react';

const ChatInterface = ({sessionId}) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const messageListRef = useRef(null)

  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [messages]);

  const API_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim() || isStreaming) return;

    const userMessage = { role: 'user', content: query };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setQuery('');
    setIsStreaming(true);
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query,
          session_id: sessionId
         }),
      });

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Read the stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          const updatedLastMessage = { ...lastMessage, content: lastMessage.content + chunk };
          return [...prev.slice(0, -1), updatedLastMessage];
        });
      }
    } catch (error) {
      console.error('Error fetching stream:', error);
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        const updatedLastMessage = { ...lastMessage, content: 'Error: Could not get a response.' };
        return [...prev.slice(0, -1), updatedLastMessage];
      });
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="message-list" ref={messageListRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          className="chat-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about the document..."
          disabled={isStreaming}
        />
        <button type="submit" className="send-button" disabled={isStreaming}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
