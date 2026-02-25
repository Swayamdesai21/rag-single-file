import { useState, useEffect } from "react";
import { uploadFile } from "../api/chatApi";

export default function FileUpload({ conversationId, activeFile, onUploadSuccess }) {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState("idle");
    const [errorMsg, setErrorMsg] = useState("");

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setStatus("idle");
            setErrorMsg("");
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        if (!conversationId) {
            setErrorMsg("Please select or create a chat first.");
            return;
        }

        setStatus("uploading");
        setErrorMsg("");

        try {
            await uploadFile(file, conversationId);
            setStatus("success");
            if (onUploadSuccess) {
                onUploadSuccess(file.name);
            }
        } catch (error) {
            console.error(error);
            setStatus("error");
            setErrorMsg(error.response?.data?.detail || "Upload failed. Please try again.");
        }
    };


    // Construct the display message
    const displayMessage = activeFile
        ? `Success! Document loaded: ${activeFile}`
        : status === "uploading"
            ? "Indexing document..."
            : errorMsg;


    return (
        <div
            className="glass"
            style={{
                padding: "0 24px",
                display: "flex",
                alignItems: "center",
                gap: "16px",
                minHeight: "72px",
                borderBottom: "1px solid var(--border)",
                zIndex: 10
            }}
        >
            <div style={{ fontWeight: "600", color: "var(--text-main)", fontSize: "14px", whiteSpace: "nowrap" }}>
                Knowledge Base
            </div>

            <div style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                gap: "12px",
                maxWidth: "600px"
            }}>
                <input
                    type="file"
                    accept=".pdf,.docx,.doc,.pptx,.ppt,.txt,.md"
                    onChange={handleFileChange}
                    style={{
                        color: "var(--text-muted)",
                        fontSize: "13px",
                        padding: "4px",
                        borderRadius: "4px",
                        backgroundColor: "rgba(0,0,0,0.2)",
                        flex: 1
                    }}
                />

                <button
                    onClick={handleUpload}
                    disabled={!file || status === "uploading"}
                    style={{
                        padding: "8px 20px",
                        borderRadius: "8px",
                        border: "none",
                        backgroundColor: status === "uploading" ? "var(--border)" : "var(--primary)",
                        color: "white",
                        cursor: !file || status === "uploading" ? "not-allowed" : "pointer",
                        fontWeight: "600",
                        fontSize: "13px",
                        boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                    }}
                >
                    {status === "uploading" ? "Indexing..." : "Upload & Index"}
                </button>
            </div>

            {displayMessage && (
                <div
                    className={status === "uploading" || status === "success" ? "loading-pulse" : "animate-fade-in"}
                    style={{
                        padding: "6px 12px",
                        borderRadius: "6px",
                        fontSize: "13px",
                        backgroundColor: activeFile ? "rgba(16, 185, 129, 0.1)" : "transparent",
                        color: activeFile ? "var(--accent)" : status === "error" ? "#ef4444" : "var(--text-muted)",
                        marginLeft: "auto",
                        fontWeight: "500",
                        border: activeFile ? "1px solid var(--accent)" : "none"
                    }}
                >
                    {displayMessage}
                </div>
            )}
        </div>
    );
}
