import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Constants
PDF_PATH    = "../data/manual.pdf"
VECTOR_DB   = "../vectorstore/_manual"
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50

def load_and_split_pdf(pdf_path):
    print(f"Loading PDF from: {pdf_path}")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks") 
    
    return chunks
def create_vector_store(chunks):
    print("Creating embeddings and vector store...")
    
    # Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
    # Create FAISS vector store from chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk
    os.makedirs(VECTOR_DB, exist_ok=True)
    vectorstore.save_local(VECTOR_DB)
    
    print(f"Vector store saved to: {VECTOR_DB}")
    return vectorstore

def main():
    print("Starting document ingestion...")
    
    # Step 1: Load and split the PDF
    chunks = load_and_split_pdf(PDF_PATH)
    
    # Step 2: Create and save vector store
    create_vector_store(chunks)
    
    print("Ingestion complete. Vector store is ready.")

if __name__ == "__main__":
    main()