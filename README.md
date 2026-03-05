## Document Question Answering System using RAG

This project implements a **Retrieval-Augmented Generation (RAG) based Document Question Answering System** using **LangChain, HuggingFace Embeddings, FAISS, and Streamlit**. The application allows users to upload a PDF document and ask questions related to its content.

The system splits the document into text chunks, converts them into embeddings, stores them in a FAISS vector database, and retrieves the most relevant chunks based on the user's query. A language model then generates answers using the retrieved context.

This project demonstrates how RAG pipelines can be used to build intelligent document search and question-answering systems.
