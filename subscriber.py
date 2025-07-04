import os
import json
import threading
from google.cloud import pubsub_v1
from dotenv import load_dotenv

# Import utility functions
from utility import (
    extract_text_from_file,
    chunk_text,
    get_embedding,
    upsert_vector_to_index,
    save_chunk_text,
    download_blob,
    delete_local_file,  # NEW: import the delete function
)

# Load environment variables from .env
load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "fall-detection-rag")
subscription_id = "fall-detection-rag-subscription"
location = os.getenv("LOCATION", "us-central1")
index_id = os.getenv("VECTOR_INDEX_ID")  # Full resource name!

# --- Chunk text mapping setup ---
CHUNK_TEXTS_FILE = "chunk_texts.json"
file_lock = threading.Lock()    

# Load existing chunk texts or initialize empty dict
if os.path.exists(CHUNK_TEXTS_FILE):
    with open(CHUNK_TEXTS_FILE, "r", encoding="utf-8") as f:
        chunk_texts = json.load(f)
else:
    chunk_texts = {}

def callback(message):
    print("Received message:")
    data = message.data.decode("utf-8")
    print(data)
    try:
        message_json = json.loads(data)
        file_name = message_json['name']  # e.g., uploads/yourfile.pdf
        bucket_name = message_json['bucket']
        local_path = download_blob(bucket_name, file_name)
        try:
            text = extract_text_from_file(local_path)
        except Exception as e:
            print(f"Error extracting text: {e}")
            delete_local_file(local_path)  # Clean up even if extraction fails
            message.ack()
            return
        print(f"Extracted text from {file_name} (first 500 chars):\n{text[:500]}...\n")
        chunks = chunk_text(text)
        print(f"Total chunks: {len(chunks)}")
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk, project_id, location)
            text_preview = chunk[:100].replace('\n', ' ')
            print(f"Chunk {idx+1}:")
            print(f"  Text: {text_preview}...")
            print(f"  Embedding (first 5 dims): {embedding[:5]}")
            metadata = {
                "document_id": file_name,
                "chunk_index": idx,
                "text_preview": text_preview
            }
            chunk_id = f"{file_name}_chunk_{idx}"
            upsert_vector_to_index(chunk_id, embedding, metadata, index_id)
            save_chunk_text(chunk_id, chunk, chunk_texts, CHUNK_TEXTS_FILE, file_lock)
        # --- DELETE LOCAL FILE AFTER PROCESSING ---
        delete_local_file(local_path)
    except Exception as e:
        print(f"Error processing message: {e}")
    message.ack()

if __name__ == "__main__":
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    print(f"Listening for messages on {subscription_path}... (Ctrl+C to exit)")
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        streaming_pull_future.result()
