from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# 1️⃣ Load PDF
loader = PyPDFLoader("paracetamol.pdf")
documents = loader.load()

# 2️⃣ Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# 3️⃣ Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4️⃣ FAISS
vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 5️⃣ Load LLM (FLAN-T5)
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# 6️⃣ Ask Question
query = "What are the side effects of paracetamol?"

# Retrieve top 3 documents
docs = retriever.invoke(query)

# Combine retrieved text
context = "\n\n".join([doc.page_content for doc in docs])

# Create prompt manually
prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

# Generate answer
result = llm.invoke(prompt)

print("\nFinal Answer:\n")
print(result)

print("\nSource Chunks:\n")
for i, doc in enumerate(docs):
    print(f"\nSource {i+1}:\n")
    print(doc.page_content)
    print("-" * 80)