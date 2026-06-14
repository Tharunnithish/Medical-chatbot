from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(data_directory):
    loader = PyPDFLoader(data_directory)
    documents = loader.load()
    return documents

def text_splitter(text_chunks):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20
    )
    chunks = text_splitter.split_documents(text_chunks)
    return chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
