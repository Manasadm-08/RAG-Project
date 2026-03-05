from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("paracetamol.pdf")
documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))

# Create Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS Vector Store
vector_store = FAISS.from_documents(chunks, embedding_model)

print("\nVector Store Created Successfully!")

# Create Retriever (Top-3)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

query = "What are the side effects of paracetamol?"

docs = retriever.invoke(query)

print("\nTop 3 Retrieved Chunks:\n")

for i, doc in enumerate(docs):
    print(f"Chunk {i+1}:\n")
    print(doc.page_content)
    print("-" * 80)