import pytesseract
import os
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    """Opens an image, converts to grayscale, and extracts text."""
    img = Image.open(image_path).convert('L')
    text = pytesseract.image_to_string(img)
    return text

def process_legal_document(text_content):
    """Splits text into chunks for the AI to read."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text_content)

def store_in_vector_db(chunks):
    """Stores text chunks into vector, and stores in a file."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./db"
    )
    return vector_db

if __name__ == "__main__":
    print("--- Testing Ingestion Pipeline ---")
    target_image = "data/test_dec28.png"
    
    try:
        raw_text = extract_text_from_image(target_image)
        print("✅ OCR Complete.")

        chunks = process_legal_document(raw_text)
        print(f"✅ Created {len(chunks)} semantic chunks.")

        store_in_vector_db(chunks)
        print("✅ Database initialized in /db folder.")

    except Exception as e:
        print(f"❌ Error: {e}")