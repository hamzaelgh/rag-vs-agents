# 🧠 AI-Powered RAG System: With & Without AI Agents 🚀  

This repository demonstrates how to build a **Retrieval-Augmented Generation (RAG) system** using **Ollama, Qdrant, BM25, and embeddings**.  
We first build a **standalone RAG pipeline** and then **extend it with AI agents** using **AutoGen and Semantic Kernel**.

---

## 📌 **Part 1: Building the Basic RAG System**

### 📝 **Features**
✅ **Hybrid Retrieval:** Combines **vector search (embeddings)** with **BM25 keyword matching**  
✅ **Multi-Language Support:** **Arabic & English queries**, using **Gemma-2B for Arabic** & **Mistral-7B for English**  
✅ **Fine-Tuned Ranking:** Uses **BM25 + embedding similarity** to improve search relevance  
✅ **Streamlit UI:** Interactive web interface for testing queries  
✅ **Local Deployment:** Runs fully **offline** using **Qdrant (Docker) and Ollama**  

---

### 🔄 **User Flow & Tool Breakdown**

| **Step**                     | **Tool Used**            | **Description** |
|------------------------------|-------------------------|----------------|
| **1. User enters a query**    | Streamlit UI            | Provides an input field for users to enter queries. |
| **2. Detect query language**  | Python (Regex)          | Determines whether the query is in Arabic or English. |
| **3. Generate query embedding** | Ollama (Mistral/Gemma) | Converts the query into a numerical vector representation. |
| **4. Retrieve relevant documents** | Qdrant (Vector DB)  | Performs a **hybrid search**: **vector similarity search** (embeddings) + **BM25 keyword match**. |
| **5. Rank retrieved documents** | BM25 (Rank-BM25)       | Ranks results based on keyword relevance and vector similarity. |
| **6. Generate AI response**   | Ollama (Mistral/Gemma)  | Uses LLM to generate an answer using the top-ranked documents as context. |
| **7. Display response**       | Streamlit UI            | Shows retrieved documents, scores, and final AI response. |

---

## 🛠️ **Setup & Installation**  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/rag-vs-agents.git
cd rag-vs-agents
```

### **2️⃣ Set Up Python Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
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
ollama pull gemma2:2b
ollama pull bge-m3
```

### **6️⃣ Prepare & Index Documents**
Store your dataset inside the `data/` folder, then run:
```bash
python basic_rag/index_documents.py
```
This will:
- Detect language (Arabic/English)
- Embed documents using **Mistral-7B (4096-dim) or Gemma-2B (2304-dim)**
- Store them in Qdrant for retrieval

### **7️⃣ Start the Streamlit UI**
```bash
streamlit run basic_rag/streamlit_app.py
```

### **8️⃣ Test Queries**
Open your browser at `http://localhost:8501` and enter any query.  
Examples:  
- **English:** `"What is artificial intelligence?"`  
- **Arabic:** `"ما هو الذكاء الاصطناعي؟"`

The system will:
1. **Auto-detect query language**
2. **Retrieve relevant documents from Qdrant**
3. **Rank results using BM25 + embedding similarity**
4. **Generate an AI response using Ollama (Mistral/Gemma)**

---

## 📌 **Part 2: Enhancing RAG with AI Agents (AutoGen & Semantic Kernel)**

### 📝 **Planned Enhancements**
✅ **Agent Collaboration:** Integrating multiple AI agents to handle retrieval, ranking, summarization, and user interaction  
✅ **AutoGen Framework:** Adding **AutoGen** to create an orchestrated multi-agent workflow  
✅ **Semantic Kernel (SK):** Using **SK planners** for more dynamic retrieval and query handling  
✅ **Human-in-the-Loop Validation:** Allowing users to correct AI responses before finalizing  
✅ **Action Execution:** Enabling AI agents to fetch external knowledge if required  

---

## 🛠️ **Planned Step-by-Step Implementation**
### **1️⃣ Install AutoGen & Semantic Kernel**
```bash
pip install pyautogen semantic-kernel
```

### **2️⃣ Define AI Agents**
We will create:
- **Retrieval Agent:** Queries Qdrant, applies ranking & re-ranking  
- **Summarization Agent:** Synthesizes results into a user-friendly response  
- **Validation Agent:** Checks output quality (optional human-in-the-loop)  

### **3️⃣ Modify Retrieval Logic**
Instead of running retrieval inside `query_rag.py`, we will use **AutoGen agents** to **decide the best retrieval method** dynamically.

### **4️⃣ Update Streamlit UI**
We will add:
- **Conversational agents:** Enabling a back-and-forth conversation  
- **User feedback mechanism:** Allowing users to upvote/downvote responses  