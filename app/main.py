from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import openai
import weaviate
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from weaviate.classes.config import Property, DataType, Configure
import io
from typing import List
import json

# Load ENV
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")

# Init FastAPI
app = FastAPI()

# CORS (allow dev usage from other origins, e.g., opening index.html directly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Weaviate (Python client v4)
parsed_url = urlparse(WEAVIATE_URL) if WEAVIATE_URL else None
http_host = parsed_url.hostname if parsed_url and parsed_url.hostname else "localhost"
http_port = parsed_url.port if parsed_url and parsed_url.port else 8080
http_secure = (parsed_url.scheme == "https") if parsed_url and parsed_url.scheme else False

# Use same host for gRPC; default OSS gRPC port is 50051.
grpc_host = http_host
grpc_port = 50051
grpc_secure = http_secure

client = weaviate.connect_to_custom(
    http_host=http_host,
    http_port=http_port,
    http_secure=http_secure,
    grpc_host=grpc_host,
    grpc_port=grpc_port,
    grpc_secure=grpc_secure,
)

# Ensure collection exists (drop and recreate for a clean start)
existing = client.collections.list_all()
if "DocumentChunk" not in existing:
    client.collections.create(
        name="DocumentChunk",
        properties=[
            Property(name="text", data_type=DataType.TEXT),
        ],
        vector_config=Configure.Vectorizer.none(),
    )

doc_collection = client.collections.get("DocumentChunk")

@app.on_event("shutdown")
def _close_weaviate_client():
    try:
        client.close()
    except Exception:
        pass


@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Basic content-type guard
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a PDF.")

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        reader = PdfReader(io.BytesIO(raw_bytes), strict=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text in PDF")

    # Chunk text with a larger window and small overlap to reduce API calls
    def chunk_text(source_text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        if chunk_size <= 0:
            return []
        chunks_local: List[str] = []
        start_index = 0
        text_len = len(source_text)
        while start_index < text_len:
            end_index = min(start_index + chunk_size, text_len)
            chunk = source_text[start_index:end_index]
            chunks_local.append(chunk)
            if end_index >= text_len:
                break
            start_index = end_index - overlap if overlap > 0 else end_index
        return chunks_local

    chunks = chunk_text(text)

    # Batch embeddings to drastically reduce round-trips
    def batched(iterable: List[str], batch_size: int = 64) -> List[List[str]]:
        return [iterable[i:i + batch_size] for i in range(0, len(iterable), batch_size)]

    total_inserted = 0
    for batch in batched(chunks):
        # Create embeddings for a batch of chunks in a single API call
        resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        # Insert each chunk with its corresponding embedding vector
        for i, datum in enumerate(resp.data):
            vector = datum.embedding
            doc_collection.data.insert({"text": batch[i]}, vector=vector)
            total_inserted += 1

    return {"message": f"Stored {total_inserted} chunks in Weaviate"}

@app.get("/query")
async def query_rag(q: str):
    # Create query embedding
    q_emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding

    # Search Weaviate (v4 collections API)
    result = doc_collection.query.near_vector(q_emb, limit=3)
    retrieved_chunks = [obj.properties["text"] for obj in result.objects]

    # GPT completion
    prompt = f"Answer based on this context:\n{retrieved_chunks}\n\nQuestion: {q}"
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": completion.choices[0].message.content,
        "context": retrieved_chunks
    }
@app.get("/query_stream")
def query_rag_stream(q: str):
    def event_stream():
        # Create query embedding
        q_emb = openai.embeddings.create(
            model="text-embedding-3-small",
            input=q
        ).data[0].embedding

        # Search Weaviate
        result = doc_collection.query.near_vector(q_emb, limit=3)
        retrieved_chunks = [obj.properties["text"] for obj in result.objects]

        # Send context first so UI can show it early
        yield f"data: {json.dumps({'type':'context','chunks': retrieved_chunks})}\n\n"

        # GPT streaming completion
        prompt = f"Answer based on this context:\n{retrieved_chunks}\n\nQuestion: {q}"
        stream = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, 'content', None)
            if delta:
                yield f"data: {json.dumps({'type':'token','content': delta})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
