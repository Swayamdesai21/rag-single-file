import { useState } from "react";
import { extractTextFromFile, uploadText, uploadFile } from "../api/chatApi";

export default function FileUpload({ conversationId, activeFile, onUploadSuccess }) {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState("idle"); // idle | extracting | uploading | success | error
    const [statusMsg, setStatusMsg] = useState("");
    const [errorMsg, setErrorMsg] = useState("");

    const handleFileChange = (e) => {
        if (e.target.files.length > 0) {
            setFile(e.target.files[0]);
            setStatus("idle");
            setStatusMsg("");
            setErrorMsg("");
        }
    };

    const handleUpload = async () => {
        if (!file || !conversationId) return;

        setErrorMsg("");
        const ext = file.name.split(".").pop().toLowerCase();

        try {
            // Step 1: Try to extract text client-side (PDF / TXT / MD)
            if (["pdf", "txt", "md"].includes(ext)) {
                setStatus("extracting");
                setStatusMsg(`Extracting text from ${file.name}...`);

                const text = await extractTextFromFile(file);

                if (!text || text.trim().length < 10) {
                    throw new Error("Could not extract readable text from this file. Try a text-based PDF.");
                }

                // Step 2: Send extracted text to backend
                setStatus("uploading");
                setStatusMsg("Indexing document...");
                await uploadText(text, file.name, conversationId);

            } else {
                // Step 2b: For DOCX/PPTX — server-side upload (usually small)
                setStatus("uploading");
                setStatusMsg("Uploading document...");
                await uploadFile(file, conversationId);
            }

            setStatus("success");
            setStatusMsg("");
            onUploadSuccess(file.name);

        } catch (error) {
            console.error("Upload error:", error);
            setStatus("error");

            let detail = "Upload failed. Please try again.";
            if (error.response?.data?.detail) {
                detail = typeof error.response.data.detail === "string"
                    ? error.response.data.detail
                    : JSON.stringify(error.response.data.detail);
            } else if (error.message) {
                detail = error.message;
            }
            setErrorMsg(detail);
        }
    };

    // Display message logic
    const displayMessage = activeFile
        ? `Success! Document loaded: ${activeFile}`
        : status === "extracting" || status === "uploading"
            ? statusMsg
            : errorMsg;

    const msgColor = activeFile
        ? { bg: "rgba(16, 185, 129, 0.1)", text: "var(--accent)", border: "1px solid var(--accent)" }
        : status === "error"
            ? { bg: "rgba(239, 68, 68, 0.1)", text: "#ef4444", border: "1px solid #ef4444" }
            : { bg: "transparent", text: "var(--text-muted)", border: "none" };

    const isProcessing = status === "extracting" || status === "uploading";

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
                    disabled={isProcessing}
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
                    disabled={!file || isProcessing}
                    style={{
                        padding: "8px 20px",
                        borderRadius: "8px",
                        border: "none",
                        backgroundColor: isProcessing ? "var(--border)" : "var(--primary)",
                        color: "white",
                        cursor: !file || isProcessing ? "not-allowed" : "pointer",
                        fontWeight: "600",
                        fontSize: "13px",
                        boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                        whiteSpace: "nowrap"
                    }}
                >
                    {status === "extracting" ? "Extracting..." : status === "uploading" ? "Indexing..." : "Upload & Index"}
                </button>
            </div>

            {displayMessage && (
                <div
                    className={isProcessing || status === "success" ? "loading-pulse" : "animate-fade-in"}
                    style={{
                        padding: "6px 12px",
                        borderRadius: "6px",
                        fontSize: "12px",
                        backgroundColor: msgColor.bg,
                        color: msgColor.text,
                        marginLeft: "auto",
                        fontWeight: "500",
                        border: msgColor.border,
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
