# Deployment Guide (Vercel)

This document explains how to deploy this RAG application to Vercel.

## Prerequisites

1.  **Qdrant Cloud Account**: Since Vercel does not have a persistent file system, you must use a hosted vector database.
    *   Sign up at [Qdrant Cloud](https://qdrant.tech/cloud/).
    *   Create a free cluster and get your **API Key** and **URL**.
2.  **Groq API Key**: Ensure you have your Groq API key ready.

## Steps to Deploy

### 1. Preparation
The codebase is already configured for Vercel using `vercel.json`. 

### 2. Environment Variables
When deploying on Vercel, add the following Environment Variables in the Vercel Dashboard:

| Variable | Description |
| :--- | :--- |
| `GROQ_API_KEY` | Your Groq API Key |
| `QDRANT_URL` | Your Qdrant Cloud Cluster URL (e.g., `https://...cloud.qdrant.io`) |
| `QDRANT_API_KEY` | Your Qdrant Cloud API Key |
| `LLM_PROVIDER` | Set to `groq` |
| `EMBEDDING_PROVIDER` | Set to `huggingface` |

### 3. Deploying
You can deploy using the Vercel CLI or by connecting your GitHub repository.

#### Using Vercel CLI:
```bash
npm install -g vercel
vercel
```

#### Using GitHub:
1. Push this code to a GitHub repository.
2. Go to Vercel Dashboard -> **Add New** -> **Project**.
3. Import your repository.
4. Set the **Root Directory** to `./` (the root of the project).
5. Add the Environment Variables listed above.
6. Click **Deploy**.

## Important Notes on Vercel
*   **Memory Limits**: Large PDF files might cause timeouts or memory issues on Vercel's free tier functions.
*   **Cold Starts**: The first request after some inactivity might take a few seconds as the Python environment and embedding models initialize.
*   **Storage**: Uploaded files in `temp_uploads` are ephemeral and will be deleted once the serverless function finishes. The data is persisted in Qdrant Cloud, which is why a remote instance is mandatory.
