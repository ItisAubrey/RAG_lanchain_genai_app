# PDF Chatbot with Groq and LangChain

This is a Streamlit-based PDF chatbot (with open source models) that allows users to upload PDF documents, which are then processed, indexed, and used to answer user queries. The app uses various components from LangChain, including PyPDFLoader, FAISS for vector storage, and OllamaEmbeddings for generating document embeddings. The chatbot interacts with a Groq model to provide responses based on the uploaded document's context.

## Features

- **PDF Upload**: Users can upload PDF files, and the app processes and splits the content into smaller chunks.
- **Document Embedding**: The app uses the `OllamaEmbeddings` to generate embeddings for the document chunks.
- **Vector Search**: FAISS is used to store and retrieve document vectors for fast similarity search.
- **Groq Integration**: The app interacts with a Groq model to generate responses based on the document's context.
- **Question Answering**: Users can ask questions, and the app will retrieve relevant context from the document to generate accurate answers.

## Requirements

To run the app, you'll need the following Python packages:

- streamlit
- langchain
- langchain-ollama
- langchain-groq
- faiss-cpu
- python-dotenv

You can install them using pip:


