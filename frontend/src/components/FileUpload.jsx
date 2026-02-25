import { useState } from "react";
import { uploadFile } from "../api/chatApi";

export default function FileUpload({ conversationId, activeFile, onUploadSuccess }) {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState("idle"); // idle, uploading, success, error
    const [errorMsg, setErrorMsg] = useState("");

    const handleFileChange = (e) => {
        if (e.target.files.length > 0) {
            setFile(e.target.files[0]);
            setStatus("idle");
            setErrorMsg("");
        }
    };

    const handleUpload = async () => {
        if (!file || !conversationId) return;

        setStatus("uploading");
        setErrorMsg("");

        try {
            await uploadFile(file, conversationId);
            setStatus("success");
            onUploadSuccess(file.name);
        } catch (error) {
            console.error("Upload error details:", error);
            setStatus("error");

            let errorDetail = "Upload failed. Please try again.";
            if (error.response?.data?.detail) {
                errorDetail = typeof error.response.data.detail === "string"
                    ? error.response.data.detail
                    : JSON.stringify(error.response.data.detail);
            } else if (error.message) {
                errorDetail = error.message;
            }

            setErrorMsg(errorDetail);
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
                        fontSize: "12px",
                        backgroundColor: activeFile ? "rgba(16, 185, 129, 0.1)" : status === "error" ? "rgba(239, 68, 68, 0.1)" : "transparent",
                        color: activeFile ? "var(--accent)" : status === "error" ? "#ef4444" : "var(--text-muted)",
                        marginLeft: "auto",
                        fontWeight: "500",
                        border: activeFile ? "1px solid var(--accent)" : status === "error" ? "1px solid #ef4444" : "none",
                        maxWidth: "400px",
                        maxHeight: "60px",
                        overflowY: "auto",
                        whiteSpace: "pre-wrap",
                        textAlign: "left"
                    }}
                >
                    {displayMessage}
                </div>
            )}
        </div>
    );
}
