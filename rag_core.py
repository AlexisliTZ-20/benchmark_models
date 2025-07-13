import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def process_pdf(pdf_path: str, pdf_id: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    db = FAISS.from_documents(docs, embeddings)

    # Guardar el vectorstore
    db.save_local(f"vectorstore/{pdf_id}")

def get_qa_chain(pdf_id: str, modelo: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local(f"vectorstore/{pdf_id}", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = Ollama(model=modelo) 
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    return chain
