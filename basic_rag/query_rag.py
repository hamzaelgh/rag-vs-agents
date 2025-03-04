from qdrant_client import QdrantClient
import ollama
import re
import numpy as np
import requests
import os
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt")

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Embedding sizes for models
EMBEDDING_SIZES = {
    "english": 1024,  # bge-m3 (Optimized for retrieval)
    "arabic": 1024,   # bge-m3 (Arabic-compatible embeddings)
}

# API Endpoints
AZURE_LANGUAGE_API_URL = "http://localhost:5000/text/analytics/v3.1/languages"

# Function to detect language using Azure AI Language API
def detect_language(text):
    """Detects query language using Azure AI Language API."""
    payload = {"documents": [{"id": "1", "text": text}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(AZURE_LANGUAGE_API_URL, json=payload, headers=headers)
        response_json = response.json()

        if "documents" in response_json and response_json["documents"]:
            detected_lang = response_json["documents"][0]["detectedLanguage"]["iso6391Name"]
            return "arabic" if detected_lang == "ar" else "english"

    except Exception as e:
        print(f"⚠️ Language detection error: {e}")

    return "english"  # Default to English if detection fails

# Function to generate embeddings
def generate_embedding(text, language):
    """Generates optimized embeddings using BGE-M3 for both Arabic & English."""
    response = ollama.embeddings(model="bge-m3", prompt=text)
    return response["embedding"], language

# Function to tokenize text for BM25
def tokenize(text):
    """Tokenizes input text for BM25 retrieval."""
    return [word.lower() for word in word_tokenize(text) if word not in string.punctuation]

# Function to search documents with optimized retrieval
def search_documents(query, language):
    """Retrieves documents using vector search & BM25 keyword matching."""
    query_vector, _ = generate_embedding(query, language)
    collection_name = "rag_docs_ar" if language == "arabic" else "rag_docs_en"

    # Ensure Qdrant collection exists
    if not client.collection_exists(collection_name):
        print(f"🚨 Collection '{collection_name}' not found in Qdrant. Skipping retrieval.")
        return []

    retrieved_docs = []

    # Perform vector search in Qdrant
    try:
        vector_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,
            with_payload=True
        )
        retrieved_docs = [{"text": hit.payload["text"], "score": hit.score} for hit in vector_results] if vector_results else []
    except Exception as e:
        print(f"⚠️ Vector search error: {e}")

    # BM25 Keyword Search Optimization for English
    if language == "english":
        try:
            corpus = [doc["text"] for doc in retrieved_docs]
            tokenized_corpus = [tokenize(doc) for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = tokenize(query)
            bm25_scores = bm25.get_scores(query_tokens)

            # Merge & rank results
            for idx, doc in enumerate(retrieved_docs):
                doc["bm25_score"] = bm25_scores[idx]

            retrieved_docs = sorted(retrieved_docs, key=lambda x: x["bm25_score"] + x["score"], reverse=True)
        except Exception as e:
            print(f"⚠️ BM25 search error: {e}")

    return retrieved_docs[:5]  # Return top 5 ranked documents

# Function to re-rank retrieved documents
def rerank_documents(query, retrieved_docs):
    """Re-ranks retrieved documents using similarity scores from BGE-M3 embeddings."""
    if not retrieved_docs:
        return ["No relevant documents found."]

    texts = [doc["text"] for doc in retrieved_docs]
    
    try:
        query_embedding = ollama.embeddings(model="bge-m3", prompt=query)["embedding"]
        doc_embeddings = [ollama.embeddings(model="bge-m3", prompt=text)["embedding"] for text in texts]

        # Compute similarity scores
        scores = [np.dot(doc_embedding, query_embedding) for doc_embedding in doc_embeddings]

        # Sort documents by highest similarity score
        ranked_docs = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)

        return [doc[0] for doc in ranked_docs[:2]]  # Return top 2 ranked documents
    except Exception as e:
        print(f"⚠️ Reranking error: {e}")
        return texts[:2]  # If reranking fails, return first 2 docs

# Function to generate AI response
def generate_response(query):
    """Generates an AI response using the retrieved context."""
    language = detect_language(query)
    llm_model = "gemma2:2b" if language == "arabic" else "qwen2.5:0.5b"

    retrieved_docs = search_documents(query, language)
    reranked_docs = rerank_documents(query, retrieved_docs)
    context = "\n".join(reranked_docs)

    system_prompt = (
        "أجب باللغة العربية فقط باستخدام المعلومات التالية:\n" + context
        if language == "arabic"
        else "Answer in English only using the following information:\n" + context
    )

    response = ollama.chat(model=llm_model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])
    
    return response["message"]["content"]

# Test queries
query_ar = "ما هو الذكاء الاصطناعي؟"
query_en = "What is artificial intelligence?"

print("🔹 Arabic Query:", query_ar)
print("🟢 Arabic Response:", generate_response(query_ar))

print("\n🔹 English Query:", query_en)
print("🟢 English Response:", generate_response(query_en))