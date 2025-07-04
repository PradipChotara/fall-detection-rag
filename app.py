import os
import json
import uuid
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from google.cloud import storage, aiplatform
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, ChatSession
from fastapi.middleware.cors import CORSMiddleware

# Import utility functions
from utility import extract_text_from_pdf, chunk_text, get_embedding, upsert_vector_to_index, save_chunk_text, download_blob

# --- Preload models at startup ---
embedding_model = None
gemini_model = None

load_dotenv()
gcs_client = storage.Client()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("LOCATION", "us-central1")
index_id = os.getenv("VECTOR_INDEX_ID")
endpoint_id = os.getenv("VECTOR_INDEX_ENDPOINT_ID")
bucket_name = os.getenv("GCS_BUCKET")
upload_folder = os.getenv("UPLOADS_FOLDER", "uploads")
deployed_index_id = os.getenv("DEPLOYED_INDEX_ID")

# Load chunk texts mapping
CHUNK_TEXTS_FILE = "chunk_texts.json"
if os.path.exists(CHUNK_TEXTS_FILE):
    with open(CHUNK_TEXTS_FILE, "r", encoding="utf-8") as f:
        chunk_texts = json.load(f)
else:
    chunk_texts = {}

# In-memory session history store (for demo/dev; use Redis/DB for production)
session_histories = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, gemini_model
    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    gemini_model = GenerativeModel("gemini-2.5-flash")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------
# get_files endpoint
# -----------------
@app.get("/get_files/")
async def get_files():
    try:
        blobs = gcs_client.list_blobs(bucket_name, prefix=upload_folder + "/")
        files = [blob.name.replace(upload_folder + "/", "") for blob in blobs if blob.name.endswith(".pdf")]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------
# Delete file endpoint
# -----------------
@app.delete("/delete_file/")
async def delete_file(filename: str = Query(..., description="Name of the file to delete")):
    try:
        # Delete file from GCS
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(f"{upload_folder}/{filename}")
        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found in GCS")
        blob.delete()

        # Delete all related chunks from vector index and chunk_texts.json
        chunk_prefix = f"{upload_folder}/{filename}_chunk_"
        chunk_ids_to_delete = [chunk_id for chunk_id in chunk_texts if chunk_id.startswith(chunk_prefix)]
        if chunk_ids_to_delete:
            aiplatform.init(project=project_id, location=location)
            index = aiplatform.MatchingEngineIndex(index_name=index_id)
            index.remove_datapoints(datapoint_ids=chunk_ids_to_delete)
            for chunk_id in chunk_ids_to_delete:
                chunk_texts.pop(chunk_id, None)
            with open(CHUNK_TEXTS_FILE, "w", encoding="utf-8") as f:
                json.dump(chunk_texts, f, ensure_ascii=False, indent=2)

        return {"status": "success", "deleted_file": filename, "deleted_chunks": chunk_ids_to_delete}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------
# Upload endpoint
# -----------------
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    temp_file_path = file.filename
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    bucket = gcs_client.bucket(bucket_name)
    destination_blob_name = f"{upload_folder}/{file.filename}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(temp_file_path)
    os.remove(temp_file_path)
    return {
        "filename": file.filename,
        "status": "uploaded to GCS",
        "gcs_path": destination_blob_name
    }

# -----------------
# Query endpoint
# -----------------
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # Client can provide, or server generates

@app.post("/query/")
async def query_qa(request: QueryRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        history = session_histories.get(session_id, [])

        # Use preloaded models
        query_embedding = embedding_model.get_embeddings([request.question])[0].values
        aiplatform.init(project=project_id, location=location)
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
        response = endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_embedding],
            num_neighbors=5,
        )
        retrieved_chunks = []
        if response and len(response) > 0 and len(response[0]) > 0:
            for neighbor in response[0]:
                chunk_id = neighbor.id
                full_text = chunk_texts.get(chunk_id, "[Full text not found]")
                similarity = getattr(neighbor, "distance", None) or getattr(neighbor, "score", None)
                retrieved_chunks.append({
                    "chunk_id": chunk_id,
                    "full_text": full_text,
                    "similarity": similarity
                })
            context = "\n\n".join([chunk["full_text"] for chunk in retrieved_chunks])
        else:
            context = ""

        history_str = ""
        for turn in history:
            history_str += f"{turn}\n"

        # --- IMPROVED PROMPT: Markdown-friendly, clear structure ---
        prompt = (
            "## Conversation History\n"
            f"{history_str}\n"
            "## Retrieved Context\n"
            f"{context}\n\n"
            f"## User Question\n{request.question}\n\n"
            "## Your Answer\n"
        )

        session = ChatSession(gemini_model)
        gemini_response = session.send_message(prompt)

        answer_text = gemini_response.text
        if not retrieved_chunks:
            answer_text += "\n\n*Note: No matching chunks were found in the knowledge base.*"

        history.append(f"User: {request.question}")
        history.append(f"Assistant: {answer_text}")
        session_histories[session_id] = history

        return {
            "answer": answer_text,
            "session_id": session_id,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
