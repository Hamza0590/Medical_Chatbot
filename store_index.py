from dotenv import load_dotenv
from pinecone import ServerlessSpec
from src.helper import load_pdf, filter_to_minimal_docs, text_split, download_embeddings_model
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

pinecone = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = pinecone


documents = load_pdf()
minimal_docs = filter_to_minimal_docs(documents)
text_chunks = text_split(minimal_docs)
embeddings_model = download_embeddings_model()

pinecone_api_key = pinecone
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
            name = index_name,
            dimension=384,
            metric="cosine",
            spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
        )
    index = pc.Index(index_name)
    docsearch = PineconeVectorStore.from_documents(
                    documents= text_chunks,
                    embedding=  embeddings_model,
                    index_name=index_name
                )


docsearch = PineconeVectorStore.from_existing_index(
            index_name= index_name,
            embedding=embeddings_model
        )
