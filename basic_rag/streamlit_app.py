import streamlit as st
from query_rag import generate_response, search_documents, detect_language
import ollama

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="RAG System", layout="wide")

st.title("🔍 RAG System - Hybrid Search with BM25 & LLMs")
st.markdown("This system retrieves relevant documents using **vector search** (Qdrant) and **BM25 keyword matching**, then generates a response using **large language models (LLMs)**.")

# ℹ️ Sidebar with Model & Retrieval Info
with st.sidebar:
    st.header("ℹ️ System Information")
    st.markdown("""
    - **Retrieval Method**: Hybrid (Vector Search + BM25)
    - **Vector Database**: [Qdrant](https://qdrant.tech/)
    - **Embedding Models**:
      - `Mistral-7B` for English (Ollama)
      - `Command-R7B-Arabic:7B` for Arabic (Ollama)
    - **Reranking**: [BGE-M3](https://huggingface.co/BAAI/bge-m3) for relevance tuning
    """)

# 🌍 Dropdown for language selection
language_option = st.selectbox("🌍 Choose Language:", ["Auto-Detect", "Arabic", "English"])

# 💬 Query input
query_text = st.text_input("💬 Enter your query:")

# 🔍 Search button
if st.button("🔎 Search"):
    if query_text:
        # 🔍 Detect or use user-selected language
        if language_option == "Auto-Detect":
            language = detect_language(query_text)
            st.info(f"🌍 **Detected Language:** `{language.capitalize()}`")
        else:
            language = "arabic" if language_option.lower() == "arabic" else "english"

        # 🔎 Perform retrieval
        retrieved_docs = search_documents(query_text, language)
        
        # Display retrieved documents
        st.subheader("📄 Retrieved Documents")
        if retrieved_docs:
            for idx, doc in enumerate(retrieved_docs):
                st.markdown(f"**{idx + 1}.** {doc['text']} _(Score: {doc['score']:.2f})_")
        else:
            st.warning("⚠️ No relevant documents found!")

        # 🤖 Generate AI response
        st.subheader("🤖 AI Response")
        ai_response = generate_response(query_text)
        st.write(ai_response)

        # 📌 Additional Debug Info (Expandable)
        with st.expander("📊 Debug Info: Retrieval & Model Selection"):
            st.markdown(f"""
            - **Selected Model**: `{ "Command-R7B-Arabic:7B" if language == "arabic" else "Mistral-7B" }`
            - **Documents Retrieved**: `{len(retrieved_docs)}` 
            - **BM25 Ranking Applied**: `✅ Yes`
            - **Vector Search**: `✅ Yes`
            """)

    else:
        st.warning("⚠️ Please enter a query!")