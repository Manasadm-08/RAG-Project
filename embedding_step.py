from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# Convert first chunk into embedding vector
vector = embedding_model.embed_query(chunks[0].page_content)

print("\nEmbedding Vector Length:", len(vector))
print("\nFirst 5 Numbers of Vector:", vector[:5])