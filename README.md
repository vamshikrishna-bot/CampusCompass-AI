# CampusCompass AI

A FastAPI college assistant backed by LangGraph, Groq, PDF retrieval, and a campus image catalog. The repository also includes standalone LangGraph examples.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Set the `GROQ_API_KEY` environment variable. Do not commit API keys or `.env` files.
4. Start the web app:

   ```bash
   uvicorn app:app --reload
   ```

The app loads `academics_handbook.pdf` and `fee_structure.pdf` at startup. Keep these PDFs and the `images/` directory in the project root.

## Deploy on Render

Create a **Python Web Service** with the project root as its root directory:

- Build command: `pip install -r requirements-render.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Environment variable: `GROQ_API_KEY` (store the value in Render's environment settings)

This installs only the web app's runtime dependencies and avoids downloading the local transformer model and FAISS.

The app builds lightweight BM25 text indexes from `academics_handbook.pdf` and `fee_structure.pdf` at startup. Keep both PDFs and the `images/` directory in the project root.
