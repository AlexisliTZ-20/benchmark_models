import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain_community.embeddings import SentenceTransformerEmbeddings

def process_pdf(pdf_path: str, pdf_id: str):
    print(f"\n[process_pdf] Iniciando procesamiento de {pdf_path}")

    t0 = time.perf_counter()
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    t1 = time.perf_counter()
    print(f" - Lectura del PDF: {len(pages)} páginas en {t1-t0:.2f} s")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,      # más grandes para menos embeddings
        chunk_overlap=60
    )

    # Chunking robusto por página + metadata de página
    docs = []
    for i, page in enumerate(pages, 1):
        page_chunks = splitter.split_documents([page])
        for c in page_chunks:
            c.metadata = {"page": i}
            docs.append(c)
    t2 = time.perf_counter()
    print(f" - Chunking: {len(docs)} chunks en {t2-t1:.2f} s")

    # Embeddings ultrarrápidos
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(" - Generando embeddings (esto es mucho más rápido)...")
    t3 = time.perf_counter()
    db = FAISS.from_documents(docs, embeddings)
    t4 = time.perf_counter()
    print(f" - Embeddings + indexado: {t4-t3:.2f} s")

    db.save_local(f"vectorstore/{pdf_id}")
    t5 = time.perf_counter()
    print(f" - Guardado del vectorstore: {t5-t4:.2f} s")
    print(f"[process_pdf] Procesamiento total: {t5-t0:.2f} s\n")

def get_qa_chain(pdf_id: str, modelo: str):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA

    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = FAISS.load_local(f"vectorstore/{pdf_id}", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model=modelo)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    return chain
