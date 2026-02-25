import { useState } from "react";

export default function ChatInput({ onSend }) {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  return (
    <div
      style={{
        width: "100%",
        maxWidth: "800px",
        margin: "0 auto",
        padding: "0 20px",
        display: "flex",
        gap: "10px",
        alignItems: "center"
      }}
    >
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask anything..."
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
        style={{
          flex: 1,
          padding: "14px 20px",
          borderRadius: "12px",
          border: "1px solid #333",
          outline: "none",
          backgroundColor: "#2a2a2a",
          color: "white",
          fontSize: "15px",
        }}
      />
      <button
        onClick={handleSend}
        style={{
          width: "44px",
          height: "44px",
          borderRadius: "10px",
          border: "none",
          backgroundColor: "#3b82f6",
          color: "white",
          fontSize: "18px",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "transform 0.1s"
        }}
        onMouseDown={(e) => e.currentTarget.style.transform = "scale(0.95)"}
        onMouseUp={(e) => e.currentTarget.style.transform = "scale(1)"}
      >
        ➤
      </button>
    </div>
  );
}


