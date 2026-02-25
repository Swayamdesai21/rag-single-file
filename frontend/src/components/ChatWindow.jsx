import { useState, useEffect, useRef } from "react";
import { sendMessage } from "../api/chatApi";
import MessageBubble from "./MessageBubble";
import ChatInput from "./ChatInput";

export default function ChatWindow({ conversationId, messages, onUpdateMessages }) {
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  const handleSend = async (text) => {
    const userMsg = { role: "user", content: text };
    const updatedWithUser = [...messages, userMsg];

    // Add a placeholder for the assistant response
    const placeholderMsg = { role: "assistant", content: "" };
    const messagesWithPlaceholder = [...updatedWithUser, placeholderMsg];

    onUpdateMessages(messagesWithPlaceholder);
    setLoading(true);

    try {
      await sendMessage(
        text,
        updatedWithUser,
        conversationId,
        (fullText) => {
          // Update the last message (the assistant's) with the streaming content
          onUpdateMessages([...updatedWithUser, { role: "assistant", content: fullText }]);
          setLoading(false); // Can turn off loading once we start receiving chunks
        }
      );
    } catch (error) {
      console.error(error);
      onUpdateMessages([
        ...updatedWithUser,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please make sure the backend is running and a file is uploaded."
        }
      ]);
      setLoading(false);
    }
  };


  const clearChat = () => {
    if (window.confirm("Are you sure you want to clear this conversation?")) {
      onUpdateMessages([]);
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);



  return (
    <div
      style={{
        flex: 1,
        width: "100%",
        position: "relative", // Root for absolute positioning
        minHeight: 0, // Allow flex container to shrink
        backgroundColor: "#1f1f1f",
      }}
    >
      {/* Clear Chat Button */}
      {messages.length > 0 && (
        <button
          onClick={clearChat}
          style={{
            position: "absolute",
            top: "20px",
            right: "20px",
            zIndex: 100,
            padding: "8px 16px",
            backgroundColor: "rgba(31, 31, 31, 0.8)",
            border: "1px solid #333",
            color: "#9ca3af",
            borderRadius: "6px",
            cursor: "pointer",
            fontSize: "12px",
            backdropFilter: "blur(4px)",
          }}
          onMouseOver={(e) => (e.target.style.borderColor = "#EF4444")}
          onMouseOut={(e) => (e.target.style.borderColor = "#333")}
        >
          Clear Chat
        </button>
      )}

      {/* Messages Scrollable Area */}
      <div
        id="chat-scroller"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: "100px", // Reserve space for the input
          overflowY: "auto",
          padding: "40px 20px",
        }}
      >
        <div
          style={{
            maxWidth: "800px",
            margin: "0 auto",
          }}
        >
          {messages.length === 0 ? (
            <div
              style={{
                textAlign: "center",
                paddingTop: "15vh",
                fontSize: "32px",
                color: "#4b5563",
                fontWeight: "300",
              }}
            >
              What can I help with?
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
              {messages.map((m, i) => (
                <MessageBubble key={i} role={m.role} content={m.content} />
              ))}
            </div>
          )}

          {loading && (
            <div style={{ padding: "10px", color: "#9ca3af", fontStyle: "italic", marginTop: "16px" }}>
              AI is thinking...
            </div>
          )}

          <div ref={bottomRef} style={{ height: "40px" }} />
        </div>
      </div>

      {/* Input Overlay Container */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: "100px",
          display: "flex",
          alignItems: "center",
          backgroundColor: "#1f1f1f",
          borderTop: "1px solid #333",
        }}
      >
        <ChatInput onSend={handleSend} />
      </div>
    </div>
  );
}
