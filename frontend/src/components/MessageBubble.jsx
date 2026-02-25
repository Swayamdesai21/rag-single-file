import ReactMarkdown from "react-markdown";

export default function MessageBubble({ role, content }) {
  const isUser = role === "user";

  return (
    <div
      className="animate-fade-in"
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: "20px",
      }}
    >
      <div
        className={isUser ? "" : "glass"}
        style={{
          maxWidth: "85%",
          padding: "16px 20px",
          borderRadius: isUser ? "20px 20px 4px 20px" : "20px 20px 20px 4px",
          backgroundColor: isUser ? "var(--primary)" : "rgba(30, 41, 59, 0.5)",
          color: "white",
          fontSize: "15px",
          lineHeight: "1.6",
          boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
          border: isUser ? "none" : "1px solid var(--border)",
        }}
      >
        {isUser ? (
          <div style={{ whiteSpace: "pre-wrap" }}>{content}</div>
        ) : (
          <div className="markdown-content">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
