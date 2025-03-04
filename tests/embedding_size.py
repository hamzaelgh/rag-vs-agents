import ollama

# Check embedding size for Gemma-2B
response_en = ollama.embeddings(model="bge-m3", prompt="test query")
print(f"bge-m3 Embedding Size: {len(response_en['embedding'])}")

# Check embedding size for qwen2.5:0.5b
response_ar = ollama.embeddings(model="bge-m3", prompt="اختبار استعلام")
print(f"bge-m3 Embedding Size: {len(response_ar['embedding'])}")