import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    """
    Opens an image, converts to grayscale, and extracts text.
    """
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

if __name__ == "__main__":
    print("--- Testing Ingestion Pipeline ---")
    target_image = "data/test_dec28.png"
    
    try:
        result = extract_text_from_image(target_image)
        print("✅ Success! Extracted Text:\n")
        print(result)
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\nTip: Ensure the file exists at: {target_image}")