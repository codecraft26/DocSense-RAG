# RAG FastAPI Demo

A minimal Retrieval-Augmented Generation (RAG) service built with FastAPI, Weaviate (v4), and OpenAI. Upload a PDF, store chunk embeddings in Weaviate, and query with semantic search; the answer is generated with OpenAI.

## Features
- PDF upload, chunking, OpenAI embeddings, Weaviate storage (v4 Collections API)
- Query endpoint that retrieves top-k chunks via vector search and asks OpenAI to answer
- Simple HTML UI to upload and query

## Requirements
- Python 3.13 (a virtual environment is provided in `venv/`)
- Docker and Docker Compose (for Weaviate)
- OpenAI API key

## Environment Variables
Create a `.env` file in the project root with:

```
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=your_org_id_optional
WEAVIATE_URL=http://localhost:8080
```

Notes:
- `WEAVIATE_URL` should point to your Weaviate instance. The included `docker-compose.yml` exposes `http://localhost:8080` and gRPC on `50051`.

## Start Weaviate
From the project root:

```bash
docker compose up -d
```

This starts Weaviate (>= 1.27.x) compatible with the Python client v4.

## Create and Activate Virtualenv
If needed (you may also use the provided `venv/`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install fastapi uvicorn weaviate-client openai PyPDF2 python-dotenv
```

## Run the API
```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

- API docs: http://127.0.0.1:8000/docs
- Root HTML UI: http://127.0.0.1:8000/

## Using the App
1. Open `http://127.0.0.1:8000/` in your browser.
2. Upload a real PDF (not a renamed file). The app extracts text and stores vectorized chunks in Weaviate.
3. Ask a question in the query box; the app retrieves the most relevant chunks and asks OpenAI to answer.

Alternatively, using curl:

```bash
# Upload via curl (multipart form)
curl -F "file=@/path/to/your.pdf" http://127.0.0.1:8000/upload_pdf

# Query
curl -G "http://127.0.0.1:8000/query" --data-urlencode "q=What is this document about?"
```

## Project Structure
```
app/
  main.py       # FastAPI app, Weaviate v4 client, endpoints
  index.html    # Simple UI for upload and query

docker-compose.yml # Weaviate (>= 1.27.x) with gRPC exposed
```

## Troubleshooting
- Uvicorn not found: activate the virtualenv or use the full path `venv/bin/uvicorn`.
- Weaviate version error: ensure Docker image is `>= 1.27.x` (compose uses a recent 1.27 tag) and port `8080` is free.
- gRPC/HTTP connectivity: compose exposes `8080` and `50051`. The client auto-derives security/ports from `WEAVIATE_URL`.
- PDF parsing errors: the API returns a 400 if the file is empty, not a PDF, corrupted, or has no extractable text.
- 405/404 from HTML: open the UI at `http://127.0.0.1:8000/` so requests hit the API origin. The UI also falls back to `http://127.0.0.1:8000` when opened from `file://`.

## Notes on Weaviate v4 Migration
- Uses `weaviate.connect_to_custom` and the Collections API.
- Vectorizer is set to `none` (`vector_config=Configure.Vectorizer.none()`); embeddings are provided from OpenAI.
- Collection `DocumentChunk` is created if missing (no destructive reset on reload).

## License
MIT
