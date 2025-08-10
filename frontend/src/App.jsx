import React, {useState} from "react";
import Header from "./components/Header";
import FileUpload from "./components/FileUpload";
import ChatInterface from "./components/ChatInterface";
import "./App.css"

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [appState, setAppState] = useState('uploading'); 
  const handleIngestSuccess = (newSessionId) => {
    setSessionId(newSessionId)
    setAppState('chatting');
  };

  return (
    <div className="app-container">
      <Header />
      <main className="main-content">
        {appState === 'uploading' ? (
          <FileUpload onUploadSuccess={handleIngestSuccess} />
        ) : (
          <ChatInterface sessionId={sessionId}/>
        )}
      </main>
    </div>
  );
}