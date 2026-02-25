import { useState, useEffect } from "react";
import ChatWindow from "./components/ChatWindow";
import FileUpload from "./components/FileUpload";
import Sidebar from "./components/Sidebar";

function App() {
  const [conversations, setConversations] = useState(() => {
    const saved = localStorage.getItem("conversations");
    return saved ? JSON.parse(saved) : [];
  });

  const [activeId, setActiveId] = useState(() => {
    return localStorage.getItem("active_conversation_id") || null;
  });

  // Persist state
  useEffect(() => {
    localStorage.setItem("conversations", JSON.stringify(conversations));
    if (activeId) {
      localStorage.setItem("active_conversation_id", activeId);
    } else {
      localStorage.removeItem("active_conversation_id");
    }
  }, [conversations, activeId]);

  const activeConversation = conversations.find((c) => c.id === activeId);

  const handleNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: `New Chat ${conversations.length + 1}`,
      messages: [],
      timestamp: Date.now()
    };
    setConversations([newChat, ...conversations]);
    setActiveId(newChat.id);
  };

  const handleUpdateMessages = (id, newMessages) => {
    setConversations(prev => prev.map(c => {
      if (c.id === id) {
        // Automatically set title if it's the first message
        let title = c.title;
        if (newMessages.length > 0 && c.messages.length === 0) {
          const firstContent = newMessages[0].content;
          title = firstContent.substring(0, 30) + (firstContent.length > 30 ? "..." : "");
        }
        return { ...c, messages: newMessages, title };
      }
      return c;
    }));
  };

  const handleUpdateFile = (fileName) => {
    if (!activeId) return;
    setConversations(prev => prev.map(c => {
      if (c.id === activeId) {
        return { ...c, activeFile: fileName };
      }
      return c;
    }));
  };

  const handleDeleteChat = (id) => {
    const filtered = conversations.filter(c => c.id !== id);
    setConversations(filtered);
    if (activeId === id) {
      setActiveId(filtered.length > 0 ? filtered[0].id : null);
    }
  };

  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        backgroundColor: "#1f1f1f",
        color: "white",
        display: "flex",
        overflow: "hidden",
      }}
    >
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100%", overflow: "hidden", minWidth: 0 }}>
        <FileUpload
          conversationId={activeId}
          activeFile={activeConversation?.activeFile || ""}
          onUploadSuccess={(name) => handleUpdateFile(name)}
        />


        {activeConversation ? (
          <ChatWindow
            conversationId={activeId}
            messages={activeConversation.messages}
            onUpdateMessages={(newMsgs) => handleUpdateMessages(activeId, newMsgs)}
          />
        ) : (


          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: "20px" }}>
            <h2 style={{ color: "#6b7280" }}>No conversation selected</h2>
            <button
              onClick={handleNewChat}
              style={{
                padding: "12px 24px",
                backgroundColor: "#3b82f6",
                border: "none",
                borderRadius: "8px",
                color: "white",
                cursor: "pointer",
                fontWeight: "600"
              }}
            >
              Start a New Chat
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


export default App;
