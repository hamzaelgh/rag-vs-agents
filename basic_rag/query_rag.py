from qdrant_client import QdrantClient
import ollama
import re
import numpy as np

import nltk
nltk.download("punkt")

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Correct embedding sizes for each model
EMBEDDING_SIZES = {
    "english": 4096,
    "arabic": 2304,
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

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import string

# Ensure NLTK tokenizer is available
nltk.download("punkt")

# Function to tokenize text for BM25
def tokenize(text):
    return [word.lower() for word in word_tokenize(text) if word not in string.punctuation]

# Function to retrieve documents using hybrid search (BM25 + Vector Search + Keyword Matches)
def search_documents(query, language):
    query_vector = generate_embedding(query, language)

    # Perform vector search in Qdrant
    try:
        vector_results = client.search(
            collection_name=f"rag_docs_{language}",
            query_vector=query_vector,
            limit=5,
            with_payload=True
        )
        retrieved_docs = [{"text": hit.payload["text"], "score": hit.score} for hit in vector_results] if vector_results else []
    except Exception as e:
        print(f"Vector search error: {e}")
        retrieved_docs = []

    # Perform keyword search (global lookup) with corrected filtering format
    try:
        keyword_results, _ = client.scroll(
            collection_name=f"rag_docs_{language}",
            scroll_filter={"must": [{"key": "text", "match": {"text": query}}]},
            limit=5,
            with_payload=True
        )
        keyword_docs = [{"text": hit.payload["text"], "score": 2.0} for hit in keyword_results]
    except Exception as e:
        print(f"Keyword search error: {e}")
        keyword_docs = []

    # Tokenize documents for BM25 scoring
    corpus = [doc["text"] for doc in keyword_docs + retrieved_docs]
    tokenized_corpus = [tokenize(doc) for doc in corpus]

    # Initialize BM25 scorer
    bm25 = BM25Okapi(tokenized_corpus)

    # Compute BM25 scores for the query
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize vector scores and BM25 scores
    max_vector_score = max([doc["score"] for doc in retrieved_docs], default=1)
    max_bm25_score = max(bm25_scores, default=1)

    for i, doc in enumerate(keyword_docs + retrieved_docs):
        bm25_score = bm25_scores[i] / max_bm25_score if max_bm25_score else 0
        vector_score = doc["score"] / max_vector_score if max_vector_score else 0

        # Weighted scoring (adjust weights as needed)
        doc["score"] = (0.5 * vector_score) + (0.3 * bm25_score) + (0.2 * 2.0)

    # Merge & sort results by final score
    final_results = sorted(keyword_docs + retrieved_docs, key=lambda x: x["score"], reverse=True)

    # Remove duplicates while preserving order
    unique_docs = []
    seen_texts = set()
    for doc in final_results:
        if doc["text"] not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(doc["text"])

    return unique_docs[:5]  # Return top 5 ranked documents

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
    llm_model = "gemma2:2b" if language == "arabic" else "mistral:7b"

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