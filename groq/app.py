import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']


st.title("PDF Chatbot")

# Allow user to upload a PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Use PyPDFLoader to process the PDF
    st.session_state.loader = PyPDFLoader("temp_uploaded_file.pdf")
    st.session_state.docs = st.session_state.loader.load()

    # Process the documents
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Make sure to specify the model parameter, e.g., "llama2" or any other available model
    st.session_state.embeddings = OllamaEmbeddings(model="llama2")  # Or use a model that you want

    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    # Setup ChatGroq
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

    # Setup prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # Create the document and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User input for questions
    prompt_text = st.text_input("Enter your question:")
    if prompt_text:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt_text})
        st.write("Response:", response['answer'])
        st.write("Response Time:", time.process_time() - start)

        with st.expander("Document Similarity Search"):
            for doc in response.get("context", []):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Clean up temporary file after processing
if os.path.exists("temp_uploaded_file.pdf"):
    os.remove("temp_uploaded_file.pdf")