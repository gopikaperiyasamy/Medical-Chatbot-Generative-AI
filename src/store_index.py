from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
import pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = str(PINECONE_API_KEY) if PINECONE_API_KEY else ""

# Initialize Pinecone client (v3 syntax)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Check existing indexes
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    # Create index if not exists
    pc.create_index(
        name=index_name,
        dimension=384,  # Replace with your model's embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

# Load and split PDF data
extracted_data = load_pdf_file(data="D:\\Medical-Chatbot-Generative-AI\\Data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Store documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

