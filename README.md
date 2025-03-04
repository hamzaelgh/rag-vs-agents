# 🧠 AI-Powered RAG System: Deployable Offline, On-Premise, or Any Cloud 🚀  

This repository demonstrates how to build a **Retrieval-Augmented Generation (RAG) system** that is **cloud-agnostic, fully deployable offline, on-premise, or on any cloud**.

The system integrates **Ollama, Qdrant, BM25, and embeddings**, with **multilingual support (Arabic & English)** and **fine-tuned retrieval enhancements** to improve accuracy and ranking.  

Additionally, **Azure AI Services** (via offline containers) enhance the system’s **language detection, document processing, content safety, and retrieval ranking**, making it robust and enterprise-ready.  

⚡ **This can help you build a PoC on Azure to assist a customer in deploying a RAG solution on-premise using Azure AI containers.**  
📌 More details: [Azure AI Containers](https://learn.microsoft.com/en-us/azure/ai-services/cognitive-services-container-support)  

---

## 📌 **Features & Capabilities**
✅ **Hybrid Retrieval:** Combines **vector search (embeddings)** with **BM25 keyword matching**  
✅ **Multi-Language Support:** **Arabic & English queries**, using **Command R7B Arabic for Arabic** & **Mistral-7B for English**  
✅ **Fine-Tuned Ranking:** Uses **BM25 + embedding similarity + reranking (bge-m3)** to enhance search relevance  
✅ **Streamlit UI:** Interactive web interface for testing queries  
✅ **Language Detection:** Uses **Azure AI Container (offline)** to detect Arabic vs. English queries  
✅ **Document Intelligence:** Uses **Azure AI Document Intelligence (offline)** for document preprocessing & OCR  
✅ **Content Moderation:** Uses **Azure AI Content Safety (offline)** to filter inappropriate content  
✅ **Local Deployment:** Runs **fully offline** using **Qdrant (Docker) and Ollama**  

---

## 🔄 **User Flow & Tool Breakdown**

| **Step**                     | **Tool Used**                 | **Description** |
|------------------------------|------------------------------|----------------|
| **1. User enters a query**    | Streamlit UI                 | Provides an input field for users to enter queries. |
| **2. Detect query language**  | Azure AI Language (Offline)  | Determines whether the query is in Arabic or English. |
| **3. Generate query embedding** | Ollama (Mistral/Command R7B Arabic) | Converts the query into a numerical vector representation. |
| **4. Retrieve relevant documents** | Qdrant (Vector DB)       | Performs a **hybrid search**: **vector similarity search** (embeddings) + **BM25 keyword match**. |
| **5. Rank retrieved documents** | BM25 (Rank-BM25) + bge-m3  | Ranks results based on keyword relevance and vector similarity. |
| **6. Generate an AI response** | Ollama (Mistral/Command R7B Arabic) | Uses LLM to generate an answer using the top-ranked documents as context. |
| **7. Apply content safety filters** | Azure AI Content Safety | Ensures the AI-generated response follows safety guidelines. |
| **8. Display response**       | Streamlit UI                 | Shows retrieved documents, scores, and final AI response. |

---

## 🛠️ **Setup & Installation**  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/rag-deployable.git
cd rag-deployable
```

### **2️⃣ Set Up Python Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Start Qdrant (Vector Database)**
Make sure **Docker** is installed, then run:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### **5️⃣ Install & Run Ollama**
Follow Ollama installation from [Ollama's official website](https://ollama.com). Then, pull the required models:
```bash
ollama pull mistral:7b
ollama pull command-r7b-arabic:7b
ollama pull bge-m3
```

### **6️⃣ Run Azure AI Containers for Language Detection & NER**
#### **Language Detection**
```bash
docker run --rm -it --platform linux/amd64 -p 5000:5000 --memory 6g --cpus 2 \
  mcr.microsoft.com/azure-cognitive-services/textanalytics/language \
  Eula=accept \
  Billing="$AZURE_LANGUAGE_BILLING_URL" \
  ApiKey="$AZURE_LANGUAGE_API_KEY"
```

#### **Named Entity Recognition (NER)**
```bash
docker run --rm -it --platform linux/amd64 -p 5001:5001 --memory 8g --cpus 1 \
  mcr.microsoft.com/azure-cognitive-services/textanalytics/ner:latest \
  Eula=accept \
  Billing="$AZURE_NER_BILLING_URL" \
  ApiKey="$AZURE_NER_API_KEY"

```



### **7️⃣ Prepare & Index Documents**
Store your dataset inside the `data/` folder, then run:
```bash
python basic_rag/index_documents.py
```
This will:
- Detect language (Arabic/English)
- Perform Named Entity Recognition (NER)
- Embed documents using **Mistral-7B (4096-dim) or Command R7B Arabic (4096-dim)**
- Store them in Qdrant for retrieval

Verify if the Qdrant Collections Exist
```
curl -X GET "http://localhost:6333/collections"
```

### **8️⃣ Start the Streamlit UI**
```bash
streamlit run basic_rag/streamlit_app.py
```

### **9️⃣ Test Queries**
Open your browser at `http://localhost:8501` and enter any query.  
Examples:  
- **English:** `"What is artificial intelligence?"`  
- **Arabic:** `"ما هو الذكاء الاصطناعي؟"`

The system will:
1. **Detect query language using Azure AI**
2. **Perform Named Entity Recognition (NER)**
3. **Retrieve relevant documents from Qdrant**
4. **Rank results using BM25 + embedding similarity + reranking**
5. **Generate an AI response using Ollama (Mistral/Command R7B Arabic)**

---

## 📌 **Addressing Arabic Language Challenges**
### **1️⃣ Challenge: Arabic Ranker Models**
📌 **Problem:** Many ranker models struggle to reconstruct answers when the supporting information is scattered across multiple chunks.  
✅ **Solution:** We integrate **BM25 + bge-m3 reranker**, which improves the ranking of relevant Arabic documents based on **semantic similarity and keyword matching**.

### **2️⃣ Challenge: Arabic Embedding Models**
📌 **Problem:** Single-word Arabic queries sometimes fail to retrieve results, even when relevant content exists in the knowledge base.  
✅ **Solution:** We use a **hybrid search approach**, combining:
   - **Vector search (Ollama embeddings)**
   - **BM25 keyword matching**
   - **Reranking using bge-m3**
   This ensures better retrieval even for **short Arabic queries**.

### **3️⃣ Mitigating those Issues** 
Our current implementation mitigates these issues with:
- Hybrid Search (BM25 + Vectors)
- Re-ranking (bge-m3)
- Named Entity Recognition (NER)
- LLM Context Expansion
---

## 🚀 **Future Improvements**
🟢 **Experiment with specialized Arabic embedding models** (e.g., Arabic-trained versions of BGE or MARBERT).  
🟢 **Optimize BM25 weights for Arabic vs. English separately** to fine-tune ranking balance.  
🟢 **Extend Named Entity Recognition (NER) to improve keyword-based lookup**.  
🟢 **Benchmark different Arabic language models for better retrieval performance**.  

---

This system is designed to be **fully deployable offline, on-premise, or in any cloud** with **optimized Arabic & English retrieval**. 🚀  

Enjoy building your **production-ready RAG system**! 🏗️🔥  
