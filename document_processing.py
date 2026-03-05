from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("paracetamol.pdf")
documents = loader.load()

print("Total Pages Loaded:", len(documents))

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total Chunks Created:", len(chunks))

print("\nFirst Chunk Preview:\n")
print(chunks[0].page_content)