import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import ollama
import re
import textwrap
import json
import csv

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Correct embedding sizes
EMBEDDING_SIZES = {
    "english": 4096,  # Mistral-7B
    "arabic": 2304,   # Gemma-2B
}

# Function to detect language
def detect_language(text):
    arabic_chars = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    return "arabic" if arabic_chars.search(text) else "english"

# Function to generate embeddings
def generate_embedding(text, lang):
    model = "gemma2:2b" if lang == "arabic" else "mistral:7b"
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Function to chunk text
def chunk_text(text, chunk_size=200):
    return textwrap.wrap(text, chunk_size)

# Function to load documents from `data/` folder
def load_documents():
    documents = []

    # Load text files
    for file in os.listdir("data"):
        file_path = os.path.join("data", file)

        if file.endswith(".txt"):
            lang = "arabic" if "arabic" in file else "english"
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
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

# Load documents from the folder
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

    for idx, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk, doc["lang"])

        point = PointStruct(
            id=int(f"{idx}"),  # Unique ID per chunk
            vector=embedding,
            payload={"text": chunk, "lang": doc["lang"]}
        )
        client.upsert(collection_name=f"rag_docs_{doc['lang']}", points=[point])

print("✅ Documents indexed successfully from `data/` folder!")