# Legal-Intel-Agent 
An autonomous RAG agent designed to automate legal discovery by synthesizing case facts from unstructured PDFs via OCR and reasoning loops.


##  Key Features
- **Agentic Reasoning:** Uses a "Plan-and-Execute" pattern to verify citations.
- **Multi-Modal Ingestion:** Tesseract OCR pipeline for "un-searchable" legal scans.
- **Self-Correction:** A dedicated "Auditor Node" that checks for hallucinations.

##  Tech Stack
- **Engine:** Python, LangChain, LangGraph
- **Models:** GPT-4o (Reasoning), LegalBERT (Embeddings)
- **Data:** Tesseract OCR, ChromaDB (Vector Store)

##  Setup
1. Clone the repo
2. Create `.env` with your `OPENAI_API_KEY`
3. Run `pip install -r requirements.txt`
4. Run `python src/ingestion.py`
