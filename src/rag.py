import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

VECTOR_DB = "../vectorstore/_manual"

def load_qa_chain():
    # Load embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vectorstore = FAISS.load_local(
        VECTOR_DB,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    prompt_template = """You are a manufacturing documentation assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I cannot find this information in the manual."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
      
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever


def ask_question(chain, retriever, question):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    
    # Show source pages
    docs = retriever.invoke(question)
    print("\nSources:")
    for doc in docs:
        page = doc.metadata.get('page', 0)
        print(f"  - Page {page + 1}: {doc.page_content[:100]}...")
    
    return answer


if __name__ == "__main__":
    print("Loading QA system...")
    chain, retriever = load_qa_chain()
    print("Ready. Type your question or 'quit' to exit.\n")
    
    while True:
        question = input("Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        ask_question(chain, retriever, question)