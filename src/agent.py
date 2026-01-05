import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
#from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#load_dotenv()

#PROMPTS

# Analyst for initial answer
GENERATION_PROMPT = """
You are a Junior Legal Analyst. Use the following context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Draft Answer:
"""

# Senior Partner for critique
CRITIQUE_PROMPT = """
You are a Senior Legal Partner. You are grading a draft answer provided by a junior analyst.
Check the following:
1. Does the answer directly address the user's question?
2. Does it cite specific facts/clauses from the context?
3. Is it accurate based *only* on the provided context?

Context: {context}
User Question: {question}
Draft Answer: {initial_answer}

If the answer is perfect, just say "PERFECT".
If the answer is missing citations or is vague, provide specific feedback on how to fix it.
Critique:
"""

# Refiner for final answer
REFINE_PROMPT = """
You are a Legal Analyst. Your previous answer was critiqued by a Senior Partner.
Please rewrite the answer to address the critique. Ensure you cite specific details from the context.

Context: {context}
User Question: {question}
Draft Answer: {initial_answer}
Senior Partner Critique: {critique}

Final Polished Answer:
"""

def get_legal_agent():
    #connect to brain (db)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory="./db", embedding_function=embeddings)
    retriever = vector_db.as_retriever()

    llm = ChatOllama(model="llama3", temperature=0)

    generate_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(GENERATION_PROMPT)
        | llm
        | StrOutputParser()
    )

    critique_chain = (
        PromptTemplate.from_template(CRITIQUE_PROMPT)
        | llm
        | StrOutputParser()
    )

    refine_chain = (
        PromptTemplate.from_template(REFINE_PROMPT)
        | llm
        | StrOutputParser()
    )

    return retriever, generate_chain, critique_chain, refine_chain

def run_agentic_loop(query):
    retriever, generate_chain, critique_chain, refine_chain = get_legal_agent()
    
    print("context...")
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])

    # Draft
    print("draft...")
    draft = generate_chain.invoke(query)
    print(f"--- DRAFT ---\n{draft}\n")

    # Critique
    print("critiquing...")
    critique = critique_chain.invoke({
        "context": context_text,
        "question": query,
        "initial_answer": draft
    })
    print(f"--- CRITIQUE ---\n{critique}\n")

    # Step 3: Decision Logic (The "Loop")
    if "PERFECT" in critique.upper():
        print("DEBUG: ✅ Draft passed checks.")
        return draft
    else:
        print("refining...")
        final_answer = refine_chain.invoke({
            "context": context_text,
            "question": query,
            "initial_answer": draft,
            "critique": critique
        })
        return final_answer

if __name__ == "__main__":

    print("\n--- Legal Intel Agent ---")
    query = input("Ask a question about the document: ")

    final_response = run_agentic_loop(query)

    print("\n--- Response ---")
    print(final_response)
