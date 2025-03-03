import os
import uuid
import json
import csv
import textwrap
import requests
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

# Load API details from environment variables (.env)
AZURE_LANGUAGE_API_URL = os.getenv("AZURE_LANGUAGE_API_URL", "http://localhost:5000/text/analytics/v3.1/languages")

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Correct embedding sizes for each model
EMBEDDING_SIZES = {
    "en": 4096,  # Mistral-7B
    "ar": 4096,  # command-r7b-arabic:7b
}

# ✅ Ensure Qdrant collections exist with correct dimensions
for lang, vector_size in EMBEDDING_SIZES.items():
    collection_name = f"rag_docs_{lang[:2]}"  # 'rag_docs_ar' for Arabic, 'rag_docs_en' for English

    if client.collection_exists(collection_name):
        print(f"🚨 Deleting incorrect collection {collection_name}")
        client.delete_collection(collection_name)  # Delete old collection

    print(f"🚀 Creating collection {collection_name} with vector size {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine"),
    )

print("✅ Qdrant collections are now correctly set up!")

# Function to detect language using Azure AI Language API
def detect_language(text):
    """Detects language using Azure AI Language API."""
    payload = {"documents": [{"id": "1", "text": text}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(AZURE_LANGUAGE_API_URL, json=payload, headers=headers)
        response_json = response.json()

        if "documents" in response_json and response_json["documents"]:
            return response_json["documents"][0]["detectedLanguage"]["iso6391Name"]  # 'ar' or 'en'
        
    except Exception as e:
        print(f"⚠️ Language detection error: {e}")

    return "en"  # Default to English if detection fails

# Function to generate embeddings using the correct LLM
def generate_embedding(text):
    """Generates embedding using the correct model based on detected language."""
    lang = detect_language(text)
    model = "command-r7b-arabic:7b" if lang == "ar" else "mistral:7b"

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"], lang

# Function to chunk text
def chunk_text(text, chunk_size=200):
    """Splits text into smaller chunks to optimize retrieval."""
    return textwrap.wrap(text, chunk_size)

# Function to load documents from `data/` folder
def load_documents():
    """Loads text, JSON, and CSV documents from the data folder."""
    documents = []

    for file in os.listdir("data"):
        file_path = os.path.join("data", file)

        # Load text files
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lang = detect_language(line.strip())
                        documents.append({"text": line.strip(), "lang": lang})

        # Load JSON files
        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                for doc in json_data:
                    lang = detect_language(doc["text"])
                    documents.append({"text": doc["text"], "lang": lang})

        # Load CSV files (assuming columns: text, lang)
        elif file.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lang = detect_language(row["text"])
                    documents.append({"text": row["text"], "lang": lang})

    return documents

# Function to index documents into Qdrant
def index_document(text):
    """Indexes a document into Qdrant with language metadata."""
    embedding, lang = generate_embedding(text)

    point = {
        "id": str(uuid.uuid4()),  # Unique UUID for each chunk
        "vector": embedding,
        "payload": {"text": text, "language": lang}
    }
    
    collection_name = f"rag_docs_{lang}"  # Store in language-specific collections
    client.upsert(collection_name=collection_name, points=[point])

# Load documents
documents = load_documents()

# Ensure Qdrant collections exist with correct dimensions
for lang, vector_size in EMBEDDING_SIZES.items():
    collection_name = f"rag_docs_{lang}"

    if client.collection_exists(collection_name):
        print(f"🚨 Deleting incorrect collection {collection_name}")
        client.delete_collection(collection_name)

    print(f"🚀 Creating collection {collection_name} with vector size {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine"),
    )

# Store document chunks in Qdrant
for doc in documents:
    chunks = chunk_text(doc["text"])

    for chunk in chunks:
        index_document(chunk)

print("✅ Documents indexed successfully from `data/` folder!")