from app.ingester import ingest_file

FILE_PATH = "/Users/desaiswayam/Desktop/L&T Groq/rag-single-file/data/R&D_Contradic_RAG.pdf"

if __name__ == "__main__":
    num_chunks = ingest_file(FILE_PATH, recreate=True)
    print(f"Successfully ingested {num_chunks} chunks.")
