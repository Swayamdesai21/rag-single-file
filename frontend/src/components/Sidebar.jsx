import React from "react";

export default function Sidebar({ conversations, activeId, onSelect, onNewChat, onDeleteChat }) {
    return (
        <div
            style={{
                width: "260px",
                height: "100%",
                backgroundColor: "#171717",
                display: "flex",
                flexDirection: "column",
                borderRight: "1px solid #333",
            }}
        >
            {/* New Chat Button */}
            <div style={{ padding: "15px" }}>
                <button
                    onClick={onNewChat}
                    style={{
                        width: "100%",
                        padding: "12px",
                        backgroundColor: "transparent",
                        color: "white",
                        border: "1px solid #444",
                        borderRadius: "6px",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: "10px",
                        fontSize: "14px",
                        transition: "background-color 0.2s",
                    }}
                    onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#2f2f2f")}
                    onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                    <span style={{ fontSize: "18px" }}>+</span> New chat
                </button>
            </div>

            {/* Conversations List */}
            <div
                style={{
                    flex: 1,
                    overflowY: "auto",
                    display: "flex",
                    flexDirection: "column",
                    gap: "2px",
                    padding: "0 10px",
                }}
            >
                {conversations.map((chat) => (
                    <div
                        key={chat.id}
                        onClick={() => onSelect(chat.id)}
                        style={{
                            padding: "10px",
                            borderRadius: "6px",
                            cursor: "pointer",
                            backgroundColor: activeId === chat.id ? "#2f2f2f" : "transparent",
                            color: activeId === chat.id ? "white" : "#9ca3af",
                            fontSize: "14px",
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            group: "true",
                        }}
                        onMouseOver={(e) => {
                            if (activeId !== chat.id) e.currentTarget.style.backgroundColor = "#242424";
                        }}
                        onMouseOut={(e) => {
                            if (activeId !== chat.id) e.currentTarget.style.backgroundColor = "transparent";
                        }}
                    >
                        <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                            {chat.title}
                        </div>
                        {activeId === chat.id && (
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteChat(chat.id);
                                }}
                                style={{
                                    background: "none",
                                    border: "none",
                                    color: "#6b7280",
                                    cursor: "pointer",
                                    padding: "4px",
                                    fontSize: "16px",
                                }}
                                onMouseOver={(e) => (e.target.style.color = "#ef4444")}
                                onMouseOut={(e) => (e.target.style.color = "#6b7280")}
                            >
                                ×
                            </button>
                        )}
                    </div>
                ))}
            </div>

            {/* Footer Info */}
            <div style={{ padding: "15px", borderTop: "1px solid #333", fontSize: "12px", color: "#666" }}>
                RAG Assistant v2.0
            </div>
        </div>
    );
}
