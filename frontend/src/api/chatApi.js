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


// ─── DYNAMIC LIBRARY LOADER ───────────────────────────────────────────────────
const PDFJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
const PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
const JSZIP_CDN = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";

let _pdfjs = null;
let _jszip = null;

async function loadScript(src, check) {
  if (check()) return;
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

async function getPdfJs() {
  if (_pdfjs) return _pdfjs;
  await loadScript(PDFJS_CDN, () => !!window.pdfjsLib);
  window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
  _pdfjs = window.pdfjsLib;
  return _pdfjs;
}

async function getJsZip() {
  if (_jszip) return _jszip;
  await loadScript(JSZIP_CDN, () => !!window.JSZip);
  _jszip = window.JSZip;
  return _jszip;
}


// ─── EXTRACTORS ───────────────────────────────────────────────────────────────

async function extractPdf(file) {
  const lib = await getPdfJs();
  const buf = await file.arrayBuffer();
  const pdf = await lib.getDocument({ data: buf }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    pages.push(content.items.map(it => it.str).join(" "));
  }
  return pages.join("\n\n");
}

async function extractDocx(file) {
  const JSZip = await getJsZip();
  const buf = await file.arrayBuffer();
  const zip = await JSZip.loadAsync(buf);

  const xmlFile = zip.file("word/document.xml");
  if (!xmlFile) throw new Error("Not a valid DOCX file");

  const xml = await xmlFile.async("string");
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const nodes = doc.getElementsByTagName("w:t");
  return Array.from(nodes).map(n => n.textContent).join(" ");
}

async function extractPptx(file) {
  const JSZip = await getJsZip();
  const buf = await file.arrayBuffer();
  const zip = await JSZip.loadAsync(buf);

  // Collect all slide XML files in order
  const slideFiles = [];
  zip.forEach((path, f) => {
    if (/^ppt\/slides\/slide\d+\.xml$/.test(path)) slideFiles.push(f);
  });

  if (slideFiles.length === 0) throw new Error("No slides found in PPTX");

  const texts = [];
  for (const slideFile of slideFiles) {
    const xml = await slideFile.async("string");
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const nodes = doc.getElementsByTagName("a:t");
    const slideText = Array.from(nodes).map(n => n.textContent).join(" ");
    if (slideText.trim()) texts.push(slideText);
  }
  return texts.join("\n\n");
}

async function extractTxt(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

/**
 * Extract text from any supported file — entirely in the browser.
 * Returns null if the file type is not supported for client-side extraction.
 */
export async function extractTextFromFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  switch (ext) {
    case "pdf": return await extractPdf(file);
    case "docx":
    case "doc": return await extractDocx(file);
    case "pptx":
    case "ppt": return await extractPptx(file);
    case "txt":
    case "md": return await extractTxt(file);
    default: return null;
  }
}


// ─── UPLOAD TEXT (send extracted text as JSON — no size limit) ────────────────
export const uploadText = async (text, filename, conversationId) => {
  const res = await axios.post(`${API_BASE_URL}/upload-text`, {
    session_id: conversationId,
    text,
    filename,
  }, {
    headers: { "Content-Type": "application/json" },
  });
  return res.data;
};


// ─── UPLOAD FILE (kept for fallback) ─────────────────────────────────────────
export const uploadFile = async (file, conversationId) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await axios.post(
    `${API_BASE_URL}/upload?session_id=${conversationId}`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return res.data;
};
