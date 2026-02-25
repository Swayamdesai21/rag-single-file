import axios from "axios";

const API_BASE_URL = import.meta.env.MODE === "production"
  ? window.location.origin
  : "http://127.0.0.1:8000";


// ─── CHAT ─────────────────────────────────────────────────────────────────────
export const sendMessage = async (question, history, conversationId = "default", onChunk) => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: conversationId, question, history }),
  });

  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    fullText += chunk;
    if (onChunk) onChunk(fullText);
  }

  return fullText;
};


// ─── TEXT EXTRACTION (Client-Side — no file size limit) ────────────────────────
const PDFJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
const PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

let pdfjsLib = null;

async function loadPdfJs() {
  if (pdfjsLib) return pdfjsLib;
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = PDFJS_CDN;
    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
      pdfjsLib = window.pdfjsLib;
      resolve(pdfjsLib);
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

async function extractPdfText(file) {
  const lib = await loadPdfJs();
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await lib.getDocument({ data: arrayBuffer }).promise;

  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map(item => item.str).join(" ");
    pages.push(pageText);
  }
  return pages.join("\n\n");
}

async function extractPlainText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

/**
 * Extract text from any supported file type — entirely in the browser.
 * For PDFs: uses PDF.js (no file size limit).
 * For TXT/MD: uses FileReader.
 */
export async function extractTextFromFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (ext === "pdf") {
    return await extractPdfText(file);
  } else if (["txt", "md"].includes(ext)) {
    return await extractPlainText(file);
  } else {
    // For DOCX/PPTX — fall back to server-side upload (these are usually small)
    return null;
  }
}


// ─── UPLOAD TEXT (send extracted text as JSON — no file size limit) ───────────
export const uploadText = async (text, filename, conversationId) => {
  const res = await axios.post(`${API_BASE_URL}/upload-text`, {
    session_id: conversationId,
    text: text,
    filename: filename,
  }, {
    headers: { "Content-Type": "application/json" },
  });
  return res.data;
};


// ─── UPLOAD FILE (original — for DOCX/PPTX or small files) ───────────────────
export const uploadFile = async (file, conversationId) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await axios.post(`${API_BASE_URL}/upload?session_id=${conversationId}`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};
