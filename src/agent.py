import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

load_dotenv()

def get_legal_agent():
    #connect to brain (db)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory="./db", embedding_function=embeddings)

    llm = ChatOllama(model="llama3", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", #"stuff" text into prompt
        retriever=vector_db.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

if __name__ == "__main__":

    agent = get_legal_agent()
    print("\n--- Legal Intel Agent Ready ---")

    query = input("Ask a question about the document: ")
    
    try:
        response = agent.invoke({"query": query})
        print("\n--- Response ---")
        print(response["result"])
    except Exception as e:
        print(f"❌ LOG ERROR: {e}")