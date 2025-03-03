import streamlit as st
from qdrant_client import QdrantClient
import ollama
import re
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is available
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
    model = "command-r7b-arabic:7b" if lang == "arabic" else "mistral:7b"
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Function to retrieve documents using hybrid search
def search_documents(query, language):
    query_vector = generate_embedding(query, language)

    # Perform vector search in Qdrant
    vector_results = client.search(
        collection_name=f"rag_docs_{language}",
        query_vector=query_vector,
        limit=5,  
        with_payload=True
    )

    retrieved_docs = [{"text": hit.payload["text"], "score": hit.score} for hit in vector_results] if vector_results else []

    # Perform keyword search (BM25)
    corpus = [doc["text"] for doc in retrieved_docs]
    tokenized_corpus = [word_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = word_tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # Merge & sort results
    for idx, doc in enumerate(retrieved_docs):
        doc["bm25_score"] = bm25_scores[idx]

    final_results = sorted(retrieved_docs, key=lambda x: x["bm25_score"] + x["score"], reverse=True)

    return final_results[:3]  # Return top 3 ranked documents

# Function to generate an LLM response
def generate_response(query):
    language = detect_language(query)
    llm_model = "command-r7b-arabic:7b" if language == "arabic" else "mistral:7b"

    retrieved_docs = search_documents(query, language)
    context = "\n".join([doc["text"] for doc in retrieved_docs])

    system_prompt = (
        "أجب باللغة العربية فقط باستخدام المعلومات التالية:\n" + context
        if language == "arabic"
        else "Answer in English only using the following information:\n" + context
    )

    response = ollama.chat(model=llm_model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])
    
    return response["message"]["content"], retrieved_docs

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="RAG System", layout="wide")

st.title("🔍 RAG System - Hybrid Search with BM25 & LLMs")
st.markdown("Enter a query to test retrieval and AI response.")

# Dropdown for language selection
language_option = st.selectbox("🌍 Choose Language:", ["Arabic", "English"])

# Query input
query_text = st.text_input("💬 Enter your query:")

# Process query on button click
if st.button("🔎 Search"):
    if query_text:
        # Convert language to lowercase for processing
        language = "arabic" if language_option.lower() == "arabic" else "english"

        # Get LLM response and retrieved documents
        ai_response, retrieved_docs = generate_response(query_text)

        # Display retrieved documents
        st.subheader("📄 Retrieved Documents")
        for idx, doc in enumerate(retrieved_docs):
            st.markdown(f"**{idx + 1}.** {doc['text']} _(Score: {doc['score']:.2f}, BM25: {doc['bm25_score']:.2f})_")

        # Display AI response
        st.subheader("🤖 AI Response")
        st.write(ai_response)
    else:
        st.warning("⚠️ Please enter a query!")