from qdrant_client import QdrantClient
import ollama
import re
import numpy as np

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

# Function to retrieve documents from Qdrant
def search_documents(query, language):
    query_vector = generate_embedding(query, language)

    results = client.search(
        collection_name=f"rag_docs_{language}",
        query_vector=query_vector,  # Changed from 'vector' to 'query_vector'
        limit=5,  # Retrieve top 5 chunks before re-ranking
        with_payload=True
    )

    if not results:
        return []

    return [{"text": hit.payload["text"], "score": hit.score} for hit in results]

# Function to re-rank retrieved documents using `bge-m3` in Ollama
def rerank_documents(query, retrieved_docs):
    if not retrieved_docs:
        return ["No relevant documents found."]
    
    texts = [doc["text"] for doc in retrieved_docs]

    # Get similarity scores using `bge-m3` in Ollama
    query_embedding = ollama.embeddings(model="bge-m3", prompt=query)["embedding"]
    doc_embeddings = [ollama.embeddings(model="bge-m3", prompt=text)["embedding"] for text in texts]

    # Compute similarity scores
    scores = [np.dot(doc_embedding, query_embedding) for doc_embedding in doc_embeddings]

    # Sort documents by highest similarity score
    ranked_docs = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
    
    return [doc[0] for doc in ranked_docs[:2]]  # Return top 2 ranked documents

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