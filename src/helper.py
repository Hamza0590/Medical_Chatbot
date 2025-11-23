from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os


def load_pdf():
    """Load PDF files from the data directory"""
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data')
    
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )   
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Filter documents to keep only page_content and source metadata"""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs


def text_split(documents):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text = text_splitter.split_documents(documents)
    return text


def download_embeddings_model():
    """Download and return the embeddings model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings_model