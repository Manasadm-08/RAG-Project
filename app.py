import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import tempfile

st.title("📄 Document Question Answering System (RAG)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF Uploaded")

    if "retriever" not in st.session_state:

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
            )

            llm = HuggingFacePipeline(pipeline=pipe)

            st.session_state.retriever = retriever
            st.session_state.llm = llm

    question = st.text_input("Ask a question")

    if question:

        with st.spinner("Generating answer..."):

            docs = st.session_state.retriever.invoke(question)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

            answer = st.session_state.llm.invoke(prompt)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Chunks")

            for i, doc in enumerate(docs):
                st.write(f"Chunk {i+1}")
                st.write(doc.page_content)
                st.write("---")