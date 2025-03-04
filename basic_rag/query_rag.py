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

# Correct embedding sizes for each model
EMBEDDING_SIZES = {
    "english": 4096,
    "arabic": 4096,  
}

# Load API details from environment variables
AZURE_LANGUAGE_API_URL = "http://localhost:5000/text/analytics/v3.1/languages"
AZURE_NER_API_URL = "http://localhost:5001/text/analytics/v3.1/entities/recognition/general"

# Function to detect language using Azure AI Language API   
def detect_language(text):
    """Calls Azure AI Language API to detect the language of the text."""
    payload = {"documents": [{"id": "1", "text": text}]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(AZURE_LANGUAGE_API_URL, json=payload, headers=headers)
    response_json = response.json()

    if "documents" in response_json and response_json["documents"]:
        detected_lang = response_json["documents"][0]["detectedLanguage"]["iso6391Name"]  # 'ar' or 'en'
        
        # Map 'ar' to 'arabic' and 'en' to 'english'
        return "arabic" if detected_lang == "ar" else "english"
    
    return "english"  # Default to English if detection fails

# # Function to extract entities using Azure AI NER API
# def extract_named_entities(text):
#     """Calls Azure AI NER API to extract named entities from the query."""
#     payload = {"documents": [{"id": "1", "text": text}]}
#     headers = {"Content-Type": "application/json"}

#     response = requests.post(AZURE_NER_API_URL, json=payload, headers=headers)
#     response_json = response.json()

#     entities = []
#     if "documents" in response_json and response_json["documents"]:
#         for entity in response_json["documents"][0]["entities"]:
#             entities.append(entity["text"])  # Extract entity names
    
#     return entities

# Function to generate embeddings
def generate_embedding(text, language):
    """Generates embedding using the correct model based on detected language."""
    model = "command-r7b-arabic:7b" if language == "arabic" else "mistral:7b"
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Function to tokenize text for BM25
def tokenize(text):
    return [word.lower() for word in word_tokenize(text) if word not in string.punctuation]

# Function to search documents using vector search, keyword matching, and NER-based lookup
# Enhance search accuracy by combining vector search, BM25, and entity matching.
def search_documents(query, language):
    """Retrieves documents using vector search, keyword matching, and NER-based lookup."""
    query_vector = generate_embedding(query, language)
    collection_name = "rag_docs_ar" if language == "arabic" else "rag_docs_en"

    # Ensure Qdrant collection exists
    if not client.collection_exists(collection_name):
        print(f"🚨 Collection '{collection_name}' not found in Qdrant. Skipping retrieval.")
        return []

    # # Extract named entities from query
    # named_entities = extract_named_entities(query)
    # print(f"🟢 Named Entities: {named_entities}")

    retrieved_docs, keyword_docs = [], []

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
        print(f"Vector search error: {e}")

    # # Perform keyword search including NER-based entity matching
    # try:
    #     keyword_results, _ = client.scroll(
    #         collection_name=collection_name,
    #         scroll_filter={"must": [{"key": "text", "match": {"value": query}}] + 
    #                        [{"key": "text", "match": {"value": entity}} for entity in named_entities]},
    #         limit=5,
    #         with_payload=True
    #     )
    #     keyword_docs = [{"text": hit.payload["text"], "score": 2.0} for hit in keyword_results]
    # except Exception as e:
    #     print(f"Keyword search error: {e}")

    # Combine and rank results
    final_results = keyword_docs + retrieved_docs
    final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)

    return final_results[:5]  # Return top 5 results

# Function to re-rank retrieved documents
def rerank_documents(query, retrieved_docs):
    if not retrieved_docs:
        return ["No relevant documents found."]
    
    # If retrieved_docs is already a list of strings, use it directly
    if isinstance(retrieved_docs[0], str):
        texts = retrieved_docs
    else:
        # Otherwise extract the text field from each document
        texts = [doc["text"] for doc in retrieved_docs]
    
    # Get similarity scores using `bge-m3` in Ollama
    try:
        query_embedding = ollama.embeddings(model="bge-m3", prompt=query)["embedding"]
        doc_embeddings = [ollama.embeddings(model="bge-m3", prompt=text)["embedding"] for text in texts]
        
        # Compute similarity scores
        scores = [np.dot(doc_embedding, query_embedding) for doc_embedding in doc_embeddings]
        
        # Sort documents by highest similarity score
        ranked_docs = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
        
        return [doc[0] for doc in ranked_docs[:2]]  # Return top 2 ranked documents
    except Exception as e:
        print(f"Reranking error: {e}")
        return texts[:2]  # If reranking fails, return first 2 docs

# Function to generate an LLM response
def generate_response(query):
    language = detect_language(query)
    llm_model = "command-r7b-arabic:7b" if language == "arabic" else "mistral:7b"

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